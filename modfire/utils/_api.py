from typing import List, Tuple, Union

from torch import nn


def findLastLinear(model: nn.Module, matchOutFeatures: int) -> List[Tuple[str, nn.Linear]]:
    matched = list()
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear) and m.out_features == matchOutFeatures:
            matched.append((n, m))
    return matched

def _getSubmoduleParent(model: nn.Module, target: str) -> nn.Module:
    if target == "":
        raise AttributeError("Can't find model itself's parent.")

    atoms: List[str] = target.split(".")
    mod: nn.Module = model
    parent: Union[nn.Module, None] = None

    for item in atoms:
        if not hasattr(mod, item):
            raise AttributeError(mod._get_name() + " has no "
                                    "attribute `" + item + "`")

        parent = mod
        mod = getattr(mod, item)

        if not isinstance(mod, nn.Module):
            raise AttributeError("`" + item + "` is not "
                                    "an nn.Module")

    if parent is None:
        raise AttributeError(f"Can't find parent given target '{target}'.")
    return parent


def replaceModule(model: nn.Module, namesAndNewModule: List[Tuple[str, nn.Module]]):
    for n, m in namesAndNewModule:
        parent = _getSubmoduleParent(model, n)
        # handle special modules
        if isinstance(parent, nn.Sequential):
            idx = int(n.split(".")[-1])
            parent.pop(idx)
            parent.insert(idx, m)
        elif isinstance(parent, nn.ModuleDict):
            key = n.split(".")[-1]
            parent.pop(key)
            parent[key] = m
        elif isinstance(parent, nn.ModuleList):
            idx = int(n.split(".")[-1])
            parent.pop(idx)
            parent.insert(idx, m)
        else:
            setattr(parent, n, m)
