import importlib

learners = {'Learner': 'models.learner'}

models = {'BasicModel': 'models.basic_model'}


def create_obj(module_name, cls_name, args):
    try:
        module = importlib.import_module(module_name)
        cls = getattr(module, cls_name)
        obj = cls(args)
        return obj
    except ImportError as e:
        exit("Unexpected error: {}".format(e))


def create_learner(config):
    learner = config.LEARNER if config.LEARNER in learners else 'Learner'
    obj = create_obj(learners[learner], learner, args=config)
    return obj


def create_model(config):
    model = config.MODEL
    if model not in models:
        exit('Unknown model {} specified'.format(model))
    obj = create_obj(models[model], model, args=config)
    return obj
