
Getting started
===============

Installation
------------

Minimum requirements:
 * Numpy
 * Scipy
 * Pandas
 * Scikit-learn

Recommended packages:
 * Py-tables (for fast persistence)
 * nltk (for various NLP features)
 * Gensim (for topic modeling)
    
Install::
    pip install ramp
or::
    cd ramp
    python setup.py install


Basic Usage
-----------
See `Getting started with Ramp: Detecting Insults <http://www.kenvanharen.com/2012/11/getting-started-with-ramp-detecting.html>`

Ramp acts on a :class:`DataContext`. DataContexts consist of 
 * a :doc:`store <stores>` that caches (and typically 
    persists) intermediate and final calculations
 * current training and preparation indices
 * the actual pandas data being analyzed (not always required)

To do anything in Ramp, you will first need to define a DataContext::
    ctx = DataContext(store='/home/test/ramp/', data=data_frame)

Once you have 
