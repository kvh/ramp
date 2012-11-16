Features
==============
Features are the core of Ramp. They are descriptions of transformations
that operate on DataFrame columns.

Things to note:

* Features attempt to store everything they compute for later reuse. They
  base cache keys on the pandas index and column name, but not the actual
  data, so for a given column name and index, a Feature will NOT recompute
  anything, even if you have changed the value inside. (This applies only in the context of a
  single storage path. Separate stores will not collide of course.)
* Features may depend on target "y" values (Feature selectors for instance).
  These features will only be built with values in the given
  `DataContext`'s train_index.
* Similarly, Features may depend on other "x" valeus.
  For instance, you normalize a column (mean zero, stdev 1) using certain rows. 
  These prepped values are stored as well so they can be used in "un-prepped"
  contexts (such as prediction on a hold out set). The given `DataContext`'s prep_index
  indicates which rows are to be used in preparation.
* Feature instances should not store state (except temporarily while being created they
  have an attached DataContext object). So the same feature object can be re-used in different
  contexts.


Creating your own features
--------------------------

Extending ramp with your own feature transformations is fairly straightforward.
Features that operate on a single feature should inherit from
`Feature`, features operating on multiple features should inherit from
`ComboFeature`. For either of these, you will need to override the `_create` method,
as well as optionally `__init__` if your feature has extra params.
Additionally, if your feature depends on other "x" values (for example it normalizes
columns using the mean and stdev of the data), you will need to define a 
`_prepare` method that returns a dict (or other picklable object)
with the required values. To get these "prepped" values, you will call
`get_prep_data` from your `_create` method. A simple (mathematically unsafe)
normalization example::

    class Normalize(Feature):

        def _prepare(self, data):
            cols = {}
            for col in data.columns:
                d = data[col]
                m = d.mean()
                s = d.std()
                cols[col] = (m, s)
            return cols

        def _create(self, data):
            col_stats = self.get_prep_data(data)
            d = DataFrame(index=data.index)
            for col in data.columns:
                m, s = col_stats[col]
                d[col] = data[col].map(lambda x: (x - m)/s)
            return d

This allows ramp to cache prep data and reuse it in contexts where the
initial data is not available, as well as prevent unnecessary recomputation.


Feature Reference
=================

.. automodule:: ramp.features.base
    :members:
    :undoc-members:

Text Features
-------------

.. automodule:: ramp.features.text
    :members:
    :undoc-members:

Combo Features
--------------

.. automodule:: ramp.features.combo
    :members:
    :undoc-members:

Trained Features
----------------

.. automodule:: ramp.features.trained
    :members:
    :undoc-members:
