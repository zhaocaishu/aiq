import importlib


def init_instance_by_config(config, **kwargs):
    module_path = config["module_path"]
    kclass = config["class"]
    cls_kwargs = config["kwargs"]

    module = importlib.import_module(module_path)
    return getattr(module, kclass)(**cls_kwargs, **kwargs)
