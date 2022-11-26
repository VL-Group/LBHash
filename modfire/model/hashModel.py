from abc import ABC, abstractmethod

from torch import nn, Tensor
from torchvision.models import get_model, get_model_weights
from vlutils.base import Registry

from modfire.utils import findLastLinear, replaceModule

from .base import BinaryWrapper


_PRETRAINED_MODEL_CLASSES = 1000


class Backbone(nn.Module):
    def __init__(self, bits: int, backbone: str = "resnet50"):
        super().__init__()
        self._backbone = get_model(backbone, weights=get_model_weights(backbone).DEFAULT)
        # modifying backbone
        lastLinears = findLastLinear(self._backbone, _PRETRAINED_MODEL_CLASSES)
        replaceModule(self._backbone, [(name, nn.Linear(linear.in_features, bits)) for name, linear in lastLinears])

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._backbone(x)
        if isinstance(outputs, tuple):
            # google-style networks, ignore aux_logits
            return outputs[0]
        else:
            return outputs


class HashRegistry(Registry):
    pass


class HashLayer(ABC, nn.Module):
    @abstractmethod
    def trainableHashFunction(self, h: Tensor, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    def forward(self, h: Tensor, *args, **kwargs) -> Tensor:
        if self.training:
            return self.trainableHashFunction(h, *args, **kwargs)
        else:
            return h.sign() > 0


@HashRegistry.register
class STEHash(HashLayer):
    def trainableHashFunction(self, h: Tensor, *_, **__) -> Tensor:
        return (h.sign() - h).detach() + h


@HashRegistry.register
class SoftHash(HashLayer):
    def trainableHashFunction(self, h: Tensor, temperature: float) -> Tensor:
        return (h / temperature).tanh()


@HashRegistry.register
class LogitHash(nn.Module):
    def trainableHashFunction(self, h: Tensor, temperature: float) -> Tensor:
        return h / temperature


class HashModel(BinaryWrapper):
    def __init__(self, bits: int, backbone: str, hashMethod: str, *args, **kwArgs):
        super().__init__(bits)
        self._backbone = Backbone(bits, backbone)
        self._hashMethod = HashRegistry.get(hashMethod)(*args, **kwArgs)

    def forward(self, x, *args, **kwArgs):
        x = self._backbone(x)
        return self._hashMethod(x, *args, **kwArgs)

    def encode(self, image: Tensor):
        h = self(image)
        return self.boolToByte(h)
