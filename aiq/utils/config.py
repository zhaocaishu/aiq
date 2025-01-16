import yaml

from easydict import EasyDict as edict


class Config(edict):
    def __init__(self, cfg: edict = None):
        if cfg is not None:
            for k, v in cfg.items():
                self._add_item(k, v)

    def __missing__(self, key):
        raise KeyError(key)

    def __getattr__(self, key):
        try:
            value = super(Config, self).__getitem__(key)
            return value
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        super(Config, self).__setitem__(key, value)

    def _add_item(self, key, value):
        if isinstance(value, dict):
            if key in self.__dict__:
                new_value = value
                value = self.__dict__[key]
                assert isinstance(value, dict)
                value.update(new_value)
            self.__setattr__(key, Config(value))
        else:
            self.__setattr__(key, value)

    def from_file(self, filename):
        """Load a config file and merge it into the default options."""
        with open(filename, "r") as f:
            try:
                cfg = yaml.safe_load(f, Loader=yaml.FullLoader)
            except:
                cfg = yaml.safe_load(f)

        self.from_config(edict(cfg))

        return self

    def from_config(self, cfg):
        assert isinstance(
            cfg, (Config, edict)
        ), "can only update dictionary or Config objects."
        for k, v in cfg.items():
            self._add_item(k, v)
        return self


# global config
config = Config()
