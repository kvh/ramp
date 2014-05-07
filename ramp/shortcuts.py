from ramp.model_definition import ModelDefinition, model_definition_factory
from ramp import modeling

def cross_validate(data=None, folds=None, repeat=1, **kwargs):
    """Shortcut to cross-validate a single configuration.

    ModelDefinition variables are passed in as keyword args, along
    with the cross-validation parameters.
    """
    md_kwargs = {}
    for arg in ModelDefinition.params:
        if arg in kwargs:
            md_kwargs[arg] = kwargs.pop(arg)
    model_def = ModelDefinition(**md_kwargs)
    return modeling.cross_validate(model_def, data, folds, repeat=repeat, **kwargs)


def cv_factory(data=None, folds=None, repeat=1, reporter_factories=[], **kwargs):
    """Shortcut to iterate and cross-validate models.
    
    All ModelDefinition kwargs should be iterables that can be
    passed to model_definition_factory.
    """
    cv_runner = kwargs.pop('cv_runner', modeling.cross_validate)
    md_kwargs = {}
    for arg in ModelDefinition.params:
        if arg in kwargs:
            md_kwargs[arg] = kwargs.pop(arg)
    model_def_fact = model_definition_factory(ModelDefinition(), **md_kwargs)
    ret = {}
    for model_def in model_def_fact:
        results, reporters = cv_runner(model_def, data, folds, repeat=repeat, reporters=[factory() for factory in reporter_factories], **kwargs)
        ret[model_def.summary] = {'model_def': model_def,
                                  'results': results,
                                  'reporters': reporters}
    
    return ret
