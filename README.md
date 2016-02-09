Odin: Organized Digital Intelligent Networks
=======

.. image:: https://readthedocs.org/projects/lasagne/badge/
    :target: http://lasagne.readthedocs.org/en/latest/

.. image:: https://travis-ci.org/Lasagne/Lasagne.svg
    :target: https://travis-ci.org/Lasagne/Lasagne

.. image:: https://img.shields.io/coveralls/Lasagne/Lasagne.svg
    :target: https://coveralls.io/r/Lasagne/Lasagne

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
    :target: https://github.com/Lasagne/Lasagne/blob/master/LICENSE

.. image:: https://zenodo.org/badge/16974/Lasagne/Lasagne.svg
   :target: https://zenodo.org/badge/latestdoi/16974/Lasagne/Lasagne

Odin is a minimalist, highly modular neural networks library, written in Python and capable of running on top of either [TensorFlow](https://github.com/tensorflow/tensorflow) or [Theano](https://github.com/Theano/Theano). It was developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research.

Use Keras if you need a deep learning library that:

- allows for easy and fast prototyping (through total modularity, minimalism, and extensibility).
- supports both convolutional networks and recurrent networks, as well as combinations of the two.
- supports arbitrary connectivity schemes (including multi-input and multi-output training).
- runs seamlessly on CPU and GPU.

Odin is compatible with: __Python 2.7-3.5__.

Its design is governed by `six principles:

* Simplicity: Be easy to use, easy to understand and easy to extend, to
  facilitate use in research
* Transparency: Do not hide Theano behind abstractions, directly process and
  return Theano expressions or Python / numpy data types
* Modularity: Allow all parts (layers, regularizers, optimizers, ...) to be
  used independently of Lasagne
* Pragmatism: Make common use cases easy, do not overrate uncommon cases
* Restraint: Do not obstruct users with features they decide not to use
* Focus: "Do one thing and do it well"


Installation
------------

Odin uses the following dependencies:

- numpy, scipy
- pyyaml
- HDF5 and h5py (optional, required if you use model saving/loading functions)
- Optional but recommended if you use CNNs: cuDNN.

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
