import abc
from enum import Enum
from functools import wraps
from typing import Any, Callable, List, Union, Dict

from vlutils.saver import Saver
from vlutils.base.freqHook import ChainHook

from modfire.config import General
from modfire.utils.registry import HookRegistry
from modfire.train.trainer import PalTrainer


class HookType(Enum):
    BeforeRunHook = "beforeRunHook"
    AfterRunHook = "afterRunHook"
    EpochStartHook = "epochStartHook"
    EpochFinishHook = "epochFinishHook"
    StepStartHook = "stepStartHook"
    StepFinishHook = "stepFinishHook"

    def __str__(self):
        return str(self.value)


def hook(hookType: HookType):
    """A decorator that marks a callable as a hook.

    NOTE: Please notice the call signature of hooks must follow example given in Usage.

    Usage:

    ```python
        # To call the hook, the function registered should follow signature below
        # where `step, epcoh` is total train steps and epochs. `trainer` is the trainer instance.
        # All hooks are called after trainer's original methods.
        # It's not recommended to alter trainer's inner states/values/attributes in hooks (Only if you really need indeed).
        # Some other arguments may be passed in. Please refer to `mcquic/train/trainer.py:L123` to see what are passed.
        @hook(HookType.xxxhook)
        def someFunction(step: int, epoch: int, trainer: PalTrainer, *args: Any, logger: Saver, **kwds: Any) -> Any:
            ...

        # For classes, since decorated function is not a instance's method, you should use `@classmethod`.
        # However, this is not recommended and has limitations.
        # Inherit from `XXXHook` is better, which is defined below.
        class SomeClass:
            ...
            @hook(HookType.xxxhook)
            @classmethod
            def someFunction(cls, step: int, epoch: int, trainer: PalTrainer, *args: Any, logger: Saver, **kwds: Any) -> Any:
                ...
    ```

    Args:
        hookType (HookType): Hook type.
    """
    def _hook(fn: Callable):
        @wraps(fn)
        def _call(*args, **kwargs):
            return fn(*args, **kwargs)
        _call.hookType = hookType
        return _call
    return _hook

"""
Implement hooks by inheriting from one or multiple hook ABCs.
Please see built-in hooks below as examples.
"""
class BeforeRunHook(abc.ABC):
    @abc.abstractmethod
    def beforeRun(self, step: int, epoch: int, trainer: PalTrainer, *args: Any, logger: Saver, **kwds: Any) -> Any:
        raise NotImplementedError
class AfterRunHook(abc.ABC):
    @abc.abstractmethod
    def afterRun(self, step: int, epoch: int, trainer: PalTrainer, *args: Any, logger: Saver, **kwds: Any) -> Any:
        raise NotImplementedError
class EpochStartHook(abc.ABC):
    @abc.abstractmethod
    def epochStart(self, step: int, epoch: int, trainer: PalTrainer, *args: Any, logger: Saver, **kwds: Any) -> Any:
        raise NotImplementedError
class EpochFinishHook(abc.ABC):
    @abc.abstractmethod
    def epochFinish(self, step: int, epoch: int, trainer: PalTrainer, *args: Any, logger: Saver, **kwds: Any) -> Any:
        raise NotImplementedError
class StepStartHook(abc.ABC):
    @abc.abstractmethod
    def stepStart(self, step: int, epoch: int, trainer: PalTrainer, *args: Any, logger: Saver, **kwds: Any) -> Any:
        raise NotImplementedError
class StepFinishHook(abc.ABC):
    @abc.abstractmethod
    def stepFinish(self, step: int, epoch: int, trainer: PalTrainer, *args: Any, logger: Saver, **kwds: Any) -> Any:
        raise NotImplementedError


# Some built-in hooks START
# Not now
# Some built-in hooks END




def splitHooks(*hooks: Union[Callable, BeforeRunHook, AfterRunHook, EpochStartHook, EpochFinishHook, StepStartHook, StepFinishHook]) -> Dict[HookType, ChainHook]:
    """Split hooks into beforeRunHook, afterRunHook, epochStartHook, epochFinishHook, stepStartHook, stepFinishHook.

    Args:
        hooks (List[Callable]): Hooks to be split.

    Returns:
        Tuple[ChainHook, ...]: Tuple of hooks.
    """
    allHooks = { k: list() for k in HookType }

    hookToTypeMap = {
        BeforeRunHook: HookType.BeforeRunHook,
        AfterRunHook: HookType.AfterRunHook,
        EpochStartHook: HookType.EpochStartHook,
        EpochFinishHook: HookType.EpochFinishHook,
        StepStartHook: HookType.StepStartHook,
        StepFinishHook: HookType.StepFinishHook
    }

    hookToCallableMap = {
        BeforeRunHook: "beforeRun",
        AfterRunHook: "afterRun",
        EpochStartHook: "epochStart",
        EpochFinishHook: "epochFinish",
        StepStartHook: "stepStart",
        StepFinishHook: "stepFinish"
    }

    for hook in hooks:
        if isinstance(hook, (BeforeRunHook, AfterRunHook, EpochStartHook, EpochFinishHook, StepStartHook, StepFinishHook)):
            for hookClass, hookType in hookToTypeMap.items():
                if isinstance(hook, hookClass):
                    allHooks[hookType].append(getattr(hook, hookToCallableMap[hookClass]))
        elif hasattr(hook, "hookType"):
            allHooks[hook.hookType].append(hook)
        else:
            raise ValueError(f"Unknown hook type of given value `{hook}`.")

    return {
        k: ChainHook(*v) for k, v in allHooks.items()
    }


def getAllBuiltinHooks() -> Dict[HookType, ChainHook]:
    raise NotImplementedError
    allHooks = list()
    for hook in BuiltInHooks.values():
        if hasattr(hook, "hookType"):
            # A decorated function. Append it directly.
            allHooks.append(hook)
        else:
            # A class-def, create it.
            allHooks.append(hook())
    return splitHooks(*allHooks)


def getAllHooks(otherHooks: List[General]) -> Dict[str, ChainHook]:
    # builtInHooks = getAllBuiltinHooks()

    otherHooksToAppend = list()
    for hook in otherHooks:
        hookFn = HookRegistry.get(hook.Key)
        if hasattr(hookFn, "hookType"):
            # A decorated function. Append it directly.
            otherHooksToAppend.append(hookFn)
        else:
            # A class-def, create it.
            otherHooksToAppend.append(hookFn(**hook.Params))
    allHooks: Dict[HookType, ChainHook] = splitHooks(*otherHooksToAppend)

    # allHooks = dict()
    # for key in builtInHooks.keys():
    #     allHooks[key] = ChainHook(builtInHooks[key], allHooks[key])
    return { str(key): value for key, value in allHooks.items() }
