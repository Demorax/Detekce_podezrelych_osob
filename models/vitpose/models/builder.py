# Copyright (c) OpenMMLab. All rights reserved.
# Fixed version for current MMCV

try:
    # Try new MMCV structure first
    from mmengine import Registry
    from mmengine.model import BaseModel


    def build_model_from_cfg(cfg, registry, default_args=None):
        """Build model from config."""
        return registry.build(cfg, default_args=default_args)


    # Create registries
    MODELS = Registry('models', build_func=build_model_from_cfg)

except ImportError:
    try:
        # Fallback to older MMCV structure
        from mmcv.utils import Registry
        from mmcv.cnn import build_model_from_cfg

        MODELS = Registry('models', build_func=build_model_from_cfg)

    except ImportError:
        # Last resort - create minimal registry
        class Registry:
            def __init__(self, name, build_func=None, parent=None):
                self.name = name
                self._module_dict = {}
                self.build_func = build_func

            def register_module(self, name=None, force=False, module=None):
                def _register(cls):
                    module_name = name if name is not None else cls.__name__
                    if module_name in self._module_dict and not force:
                        raise KeyError(f'{module_name} is already registered')
                    self._module_dict[module_name] = cls
                    return cls

                if module is not None:
                    return _register(module)
                return _register

            def build(self, cfg, default_args=None):
                if isinstance(cfg, dict):
                    cfg_copy = cfg.copy()
                    obj_type = cfg_copy.pop('type')
                    if obj_type in self._module_dict:
                        obj_cls = self._module_dict[obj_type]
                        return obj_cls(**cfg_copy)
                    else:
                        raise KeyError(f'{obj_type} is not in the registry')
                raise TypeError('cfg must be a dict')


        def build_model_from_cfg(cfg, registry, default_args=None):
            return registry.build(cfg, default_args)


        MODELS = Registry('models', build_func=build_model_from_cfg)

# Create aliases for different components
BACKBONES = MODELS
NECKS = MODELS
HEADS = MODELS
LOSSES = MODELS
POSENETS = MODELS
MESH_MODELS = MODELS


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)


def build_posenet(cfg):
    """Build posenet."""
    return POSENETS.build(cfg)


def build_mesh_model(cfg):
    """Build mesh model."""
    return MESH_MODELS.build(cfg)