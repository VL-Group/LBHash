from warnings import warn
from typing import Callable,Any

from torch import nn
from torch.optim import Adam, SGD, Optimizer
from torch.optim.lr_scheduler import _LRScheduler, ExponentialLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau
try:
    from apex.optimizers import FusedLAMB
    FusedLAMB = FusedLAMB
except ImportError:
    FusedLAMB = None
    warn("Couldn't import apex's `FusedLAMB` optimizer, ignored in `OptimRegistry`.")
from vlutils.base import Registry


__all__ = [
    "ModelRegistry",
    "SchdrRegistry",
    "CriterionRegistry",
    "OptimRegistry",
    "CriterionRegistry",
    "HookRegistry",
    "FunctionRegistry"
]


class OptimRegistry(Registry[Callable[..., Optimizer]]):
    pass

class SchdrRegistry(Registry[Callable[..., _LRScheduler]]):
    pass

class HookRegistry(Registry[Any]):
    pass

class CriterionRegistry(Registry[Callable[..., nn.Module]]):
    pass

class ModelRegistry(Registry[Callable[..., nn.Module]]):
    pass

class FunctionRegistry(Registry[Callable]):
    pass


OptimRegistry.register("Adam")(Adam)
OptimRegistry.register("SGD")(SGD)
if FusedLAMB is not None:
    OptimRegistry.register("FusedLAMB")(FusedLAMB)

SchdrRegistry.register("ExponentialLR")(ExponentialLR)
SchdrRegistry.register("CosineAnnealingWarmRestarts")(CosineAnnealingWarmRestarts)
SchdrRegistry.register("ReduceLROnPlateau")(ReduceLROnPlateau)
