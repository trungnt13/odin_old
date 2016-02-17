Odin: Organized Digital Intelligent Networks
=======

[![image](https://readthedocs.org/projects/lasagne/badge/)](http://lasagne.readthedocs.org/en/latest/)
[![image](https://travis-ci.org/Lasagne/Lasagne.svg)](https://travis-ci.org/Lasagne/Lasagne)
[![image](https://img.shields.io/coveralls/Lasagne/Lasagne.svg)](https://coveralls.io/r/Lasagne/Lasagne)

Odin contains a number of ultilities to accelerate the development of neural network using existing libraries, includes: keras and Lasagne. The library is written in Python supported constructing computational graph on top of either [Theano](https://github.com/Theano/Theano) or [TensorFlow](https://github.com/tensorflow/tensorflow). 

Our goal is creating minimalism and flexible design to speed up the overall pipeline. By simplifying the understand of Artificial Neural Networks as a "nested composition of funcitons", our library concentrate on advancing 4 components:

- Accelerate features preprocessing using MapReduce based MPI to distribute the computation of large dataset.
- High performance data manipulation using [h5py](http://docs.h5py.org/en/latest/index.html), provides high level of abstraction with extremely fast batching and shuffling data during training.
- A general purpose training procudure which optimizes any differentiable functions.
- Mechanism to store model from any frameworks as creator function.

Odin is compatible with: __Python 2.7-3.5__.

Installation
------------

Odin uses the following dependencies:

- numpy, scipy
- theano, tensorflow or both
- HDF5 and h5py (optional, required if you use model/dataset saving/loading)
- matplotlib for visualization library

*When using the Theano backend:*

- Theano
    - [See installation instructions](http://deeplearning.net/software/theano/install.html#install).

**Note**: You should use the latest version of Theano, not the PyPI version. Install it with:
```
sudo pip install git+git://github.com/Theano/Theano.git
```

*When using the TensorFlow backend:*

- TensorFlow
    - [See installation instructions](https://github.com/tensorflow/tensorflow#download-and-setup).

To install Odin, `cd` to the Odin folder and run the install command:
```
sudo python setup.py install
```

You can also install Odin from PyPI:
```
sudo pip install odin
```

Documentation
-------------

<Under development>


Example
-------

<Under development>

Development
-----------

Odin is a work in progress, input is welcome.
