
Welcome to ramp's documentation!
================================

Ramp is a python package for rapid machine learning prototyping.
It provides a simple, declarative syntax
for exploring features, algorithms and transformations quickly and efficiently.
At its core it's a unified `pandas <http://pandas.pydata.org>`_-based framework for working with
existing python machine learning and statistics libraries (scikit-learn, rpy2, etc.).

Features
^^^^^^^^

* Fast caching and persistence of all intermediate and final calculations -- nothing is recomputed unnecessarily.
* Advanced training and preparation logic. Ramp respects the current training set, even when using complex trained features and blended predictions, and also tracks the given preparation set (the x values used in feature preparation -- e.g. the mean and stdev used for feature normalization.)
* A growing library of feature transformations, metrics and estimators. Ramp's simple API allows for easy extension.

Contents:

.. toctree::
   :maxdepth: 2

   intro
   Data contexts <context>
   configurations
   features
   stores
   estimators
   reporters


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

