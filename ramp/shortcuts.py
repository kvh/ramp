import numpy as np
from prettytable import PrettyTable, ALL

from ramp.model_definition import ModelDefinition, ModelDefinitionFactory
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


def cv_factory(data=None, folds=None, repeat=1, **kwargs):
    """Shortcut to iterate and cross-validate models.

    All ModelDefinition kwargs should be iterables that can be
    passed to a ModelDefinitionFactory.
    """
    cv_runner = kwargs.pop('cv_runner', modeling.cross_validate)
    md_kwargs = {}
    for arg in ModelDefinition.params:
        if arg in kwargs:
            md_kwargs[arg] = kwargs.pop(arg)
    model_def_fact = ModelDefinitionFactory(ModelDefinition(), **md_kwargs)
    all_results = []
    for model_def in model_def_fact:
        results, metrics, reports = cv_runner(model_def, data, folds, repeat=repeat, **kwargs)
        #TODO
    #TODO


    # for conf in fact:
    #     ctx = DataContext(store, data)
    #     results.append(cv(conf, ctx, **fargs))
    # t = PrettyTable(["Configuration", "Score"])
    # t.hrules = ALL
    # t.align["Configuration"] = "l"
    # for r in results:
    #     scores_dict = r['scores']
    #     s = ""
    #     for metric, scores in scores_dict.items():
    #         s += "%s: %s" % (metric, pprint_scores(scores))
    #     t.add_row([str(r['config']), s])
    # print t
    # return ctx
