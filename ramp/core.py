registry = {}
def _register_dataset(ds):
    if ds.name in registry:
        raise KeyError("dataset '%s' already registered"%ds.name)
    registry[ds.name] = ds

def get_dataset(name):
    if name not in registry:
        raise KeyError("dataset '%s' has not been registered"%name)
    return registry[name]


