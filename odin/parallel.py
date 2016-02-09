from __future__ import print_function, division

import numpy as np

# ======================================================================
# Multiprocessing
# ======================================================================
class mpi():

    """docstring for mpi"""
    @staticmethod
    def segment_job(file_list, n_seg):
        '''
        Example
        -------
            >>> segment_job([1,2,3,4,5],2)
            >>> [[1, 2, 3], [4, 5]]
            >>> segment_job([1,2,3,4,5],4)
            >>> [[1], [2], [3], [4, 5]]
        '''
        # by floor, make sure and process has it own job
        size = int(np.ceil(len(file_list) / float(n_seg)))
        if size * n_seg - len(file_list) > size:
            size = int(np.floor(len(file_list) / float(n_seg)))
        # start segmenting
        segments = []
        for i in xrange(n_seg):
            start = i * size
            if i < n_seg - 1:
                end = start + size
            else:
                end = max(start + size, len(file_list))
            segments.append(file_list[start:end])
        return segments

    @staticmethod
    def div_n_con(path, file_list, n_job, div_func, con_func):
        ''' Divide and conquer strategy for multiprocessing.
        Parameters
        ----------
        path : str
            path to save the result, all temp file save to path0, path1, path2...
        file_list : list
            list of all file or all job to do processing
        n_job : int
            number of processes
        div_func : function(save_path, jobs_list)
            divide function, execute for each partition of job
        con_func : function(save_path, temp_paths)
            function to merge all the result
        Returns
        -------
        return : list(Process)
            div_processes and con_processes
        '''
        import multiprocessing
        job_list = mpi.segment_job(file_list, n_job)
        file_path = [path + str(i) for i in xrange(n_job)]
        div_processes = [multiprocessing.Process(target=div_func, args=(file_path[i], job_list[i])) for i in xrange(n_job)]
        con_processes = multiprocessing.Process(target=con_func, args=(path, file_path))
        return div_processes, con_processes

    @staticmethod
    def preprocess_mpi(jobs_list, features_func, save_func, n_cache=30, log_point=50):
        ''' Wrapped preprocessing procedure in MPI.
                    root
                / / / | \ \ \
                features_func
                \ \ \ | / / /
                  save_func
            * NO need call Barrier at the end of this methods

        Parameters
        ----------
        jobs_list : list
            [data_concern_job_1, job_2, ....]
        features_func : function(job_i)
            function object to extract feature from each job
        save_func : function([job_i,...])
            transfer all data to process 0 as a list for saving to disk
        n_cache : int
            maximum number of cache for each process before gathering the data
        log_point : int
            after this amount of preprocessed data, print log

        Notes
        -----
        Any None return by features_func will be ignored

        Example
        -------
        >>> jobs = range(1, 110)
        >>> if rank == 0:
        >>>     f = h5py.File('tmp.hdf5', 'w')
        >>>     idx = 0
        >>> def feature_extract(j):
        >>>     return rank
        >>> def save(j):
        >>>     global idx
        >>>     f[str(idx)] = str(j)
        >>>     idx += 1
        >>> dnntoolkit.mpi.preprocess_mpi(jobs, feature_extract, save, n_cache=5)
        >>> if rank == 0:
        >>>     f['idx'] = idx
        >>>     f.close()
        '''
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        npro = comm.Get_size()

        #####################################
        # 1. Scatter jobs for all process.
        if rank == 0:
            logger.info('Process 0 found %d jobs' % len(jobs_list))
            jobs = mpi.segment_job(jobs_list, npro)
            n_loop = max([len(i) for i in jobs])
        else:
            jobs = None
            n_loop = 0
            logger.info('Process %d waiting for Process 0!' % rank)
        comm.Barrier()

        jobs = comm.scatter(jobs, root=0)
        n_loop = comm.bcast(n_loop, root=0)
        logger.info('Process %d receive %d jobs' % (rank, len(jobs)))

        #####################################
        # 2. Start preprocessing.
        data = []

        for i in xrange(n_loop):
            if i % n_cache == 0 and i > 0:
                all_data = comm.gather(data, root=0)
                if rank == 0:
                    logger.info('Saving data at process 0')
                    all_data = [k for j in all_data for k in j]
                    if len(all_data) > 0:
                        save_func(all_data)
                data = []

            if i >= len(jobs): continue
            feature = features_func(jobs[i])
            if feature is not None:
                data.append(feature)

            if i % log_point == 0:
                logger.info('Rank:%d preprocessed %d files!' % (rank, i))

        all_data = comm.gather(data, root=0)
        if rank == 0:
            logger.info('Finished preprocess_mpi !!!!\n')
            all_data = [k for j in all_data for k in j]
            if len(all_data) > 0:
                save_func(all_data)
