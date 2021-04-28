from .defaults import _C as cfg_default


def get_cfg_default():
    return cfg_default.clone()

### this package privides cfg which is a basic setting of trainer
### and this basic setting can be modified by da/configs/.yaml file