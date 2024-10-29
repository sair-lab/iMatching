import inspect
import warnings
from dataclasses import field, make_dataclass
from typing import Any, Dict, List, Tuple, Type, TypeVar

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from omegaconf.base import is_valid_value_annotation

T = TypeVar("T")


_config_node_registry: Dict[Type[T], Tuple[str, str]] = {}


def hydra_config(name: str, group: str = None, package: str = None):
    """Generates a pseudo-config node equivalent to a `group/name.yaml` with content:
    ```yaml
        @package: <package>
        <key0>: <value0>
        ...
    ```
    """
    def _do_register(cls: Type[T]):
        _config_node_registry[cls] = (group, name)
        ConfigStore.instance().store(name=name, group=group, package=package, node=cls)
        return cls
    return _do_register


def hydra_instantiable(name: str, group: str = None, package: str = None, override_group: Dict[str, str] = None, **kwargs):
    """Generates a pseudo-config node that can be instantiated with `hydra.utils.instantiate()`.

    The generated node is equivalent to a `group/name_gencfg.yaml` with content:
    ```yaml
        @package: <package>
        _target_: <package.module.class>
        <kwarg0>: <kwarg0_value>
        ...
    ```
    kwargs of the original `__init__` is included but overriden by `kwargs` parameter.
    """
    def _do_register(cls: Type[T]):
        # get default kwargs (and values) of cls.__init__
        init_args = {}

        def _add_filed(name, anno, default):
            # infer type if doable
            anno = type(default) if anno is None and default is not None else anno
            # use factory method for mutable fields
            if default.__class__.__hash__ is None:
                _field = field(default_factory=lambda: default)
            else:
                _field = field(default=default)
            # only override default values
            if name in init_args:
                orig_args = init_args[name]
                init_args[name] = (orig_args[0], anno if anno != Any else orig_args[1], _field)
            else:
                init_args[name] = (name, anno, _field)

        signature = inspect.signature(cls.__init__)
        annotation = cls.__init__.__annotations__
        _add_filed("_target_", str, f"{cls.__module__}.{cls.__qualname__}")
        default_list = []
        for k, v in signature.parameters.items():
            anno = annotation[k] if k in annotation else Any

            # if config class is registered, use default list to compose
            if anno in _config_node_registry:
                key = k
                reg_group, reg_name = _config_node_registry[anno]
                # migrate registered config under the current parameter
                if override_group is not None and k in override_group:
                    ovr_group = override_group[k]
                    key = f"{ovr_group}@{k}"
                elif f"{group}/{k}" != reg_group:
                    warnings.warn(f"Config '{reg_name}' registered under '{reg_group}' is used for parameter '{k}' of "
                                  f"initializable '{group}/{name}'.")
                default_list.append({key: reg_name})
            else:
                if not is_valid_value_annotation(anno):
                    anno = Any

                if k == "self":
                    continue
                elif v.default is not inspect.Parameter.empty:
                    # has default value from cls.__init__
                    default = v.default
                else:
                    default = MISSING
                _add_filed(k, anno, default)

        # add default list
        # Any: omegaconf doesn't accept list type
        if len(default_list) > 0:
            _add_filed("defaults", List, default_list)

        # override by kwargs
        for k, v in kwargs.items():
            _add_filed(k, Any, v)

        # create ad-hoc dataclass package_module_name
        _dataclass = make_dataclass(f"{group.replace('/', '.')}.{name}_gencfg".replace("-", "_"), list(init_args.values()))
        ConfigStore.instance().store(name=f"{name}", group=group, package=package, node=_dataclass)
        return cls

    return _do_register
