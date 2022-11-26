import abc
from typing import Optional

from rich.progress import Progress
import torch
from torch import nn
from torch.utils.data import IterDataPipe
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService

from .searcher import BinarySearcher, PQSearcher


class BaseWrapper(nn.Module, abc.ABC):
    _dummy: torch.Tensor
    def __init__(self):
        super().__init__()
        self.register_buffer("_dummy", torch.empty([1]), persistent=False)
    @abc.abstractmethod
    def encode(self, image: torch.Tensor):
        raise NotImplementedError
    @abc.abstractmethod
    def add(self, images: IterDataPipe, progress: Optional[Progress] = None):
        raise NotImplementedError
    @abc.abstractmethod
    def search(self, queries: IterDataPipe, numReturns: int, progress: Optional[Progress] = None) -> torch.Tensor:
        raise NotImplementedError


class BinaryWrapper(BaseWrapper):
    _byteTemplate: torch.Tensor
    def __init__(self, bits: int):
        super().__init__()
        self.database = BinarySearcher(bits)
        self.register_buffer("_byteTemplate", torch.tensor([int(2 ** x) for x in range(8)]))

    def boolToByte(self, x: torch.Tensor) -> torch.Tensor:
        """Convert D-dim bool tensor to byte tensor along the last dimension.

        Args:
            x (torch.Tensor): [..., D], D-dim bool tensor.

        Returns:
            torch.Tensor: [..., D // 8], Converted byte tensor.
        """
        return (x.reshape(*x.shape[:-1], -1, 8) * self._byteTemplate).sum(-1).byte()

    @torch.no_grad()
    def add(self, images: IterDataPipe, progress: Optional[Progress] = None):
        if progress is not None:
            task = progress.add_task(f"[ Index ]", total=None, progress=f"{0:4d} images", suffix="")
        dataLoader = DataLoader2(images, reading_service=MultiProcessingReadingService(num_workers=8, pin_memory=True))
        allFeatures = list()
        total = 0
        for image in dataLoader:
            # [N, D]
            h = self.encode(image.to(self._dummy.device, non_blocking=True))
            allFeatures.append(h.cpu())
            total += len(h)
            if progress is not None:
                progress.update(task, advance=1, progress=f"{total:4d} images")
        # [N, D]
        allFeatures = torch.cat(allFeatures)
        return self.database.add(allFeatures.numpy())

    @torch.no_grad()
    def search(self, queries: IterDataPipe, numReturns: int, progress: Optional[Progress] = None) -> torch.Tensor:
        if progress is not None:
            task = progress.add_task(f"[ Query ]", total=None, progress=f"{0:4d} queries", suffix="")
        dataLoader = DataLoader2(queries, reading_service=MultiProcessingReadingService(num_workers=8, pin_memory=True))
        allFeatures = list()
        total = 0
        for image in dataLoader:
            # [D]
            h = self.encode(image.to(self._dummy.device, non_blocking=True))
            allFeatures.append(h.cpu())
            total += len(h)
            if progress is not None:
                progress.update(task, advance=1, progress=f"{total:4d} queries")
        # [N, D]
        allFeatures = torch.cat(allFeatures)
        return torch.from_numpy(self.database.search(allFeatures.numpy(), numReturns))


class PQWrapper(BaseWrapper):
    codebook: nn.Parameter
    def __init__(self, m: int, k: int, d: int):
        super().__init__()
        self.codebook = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(m, k, d // m)))
        self.database = PQSearcher(self.codebook .cpu().numpy())

    def updateCodebook(self):
        self.database.assignCodebook(self.codebook.cpu().numpy())

    def eval(self):
        self.updateCodebook()
        return super().eval()

    @torch.no_grad()
    def add(self, images: IterDataPipe, progress: Optional[Progress] = None):
        if progress is not None:
            task = progress.add_task(f"[ Index ]", total=None, progress=f"{0:4d} images", suffix="")
        dataLoader = DataLoader2(images, reading_service=MultiProcessingReadingService(num_workers=8, pin_memory=True))
        allFeatures = list()
        total = 0
        for image in dataLoader:
            # [N, D]
            x = self.encode(image.to(self._dummy.device, non_blocking=True))
            allFeatures.append(x.cpu())
            total += len(x)
            if progress is not None:
                progress.update(task, advance=1, progress=f"{total:4d} images")
        # [N, D]
        allFeatures = torch.cat(allFeatures)
        return self.database.add(allFeatures.numpy())

    @torch.no_grad()
    def search(self, queries: IterDataPipe, numReturns: int, progress: Optional[Progress] = None) -> torch.Tensor:
        if progress is not None:
            task = progress.add_task(f"[ Query ]", total=None, progress=f"{0:4d} queries", suffix="")
        dataLoader = DataLoader2(queries, reading_service=MultiProcessingReadingService(num_workers=8, pin_memory=True))
        allFeatures = list()
        total = 0
        for image in dataLoader:
            # [D]
            x = self.encode(image.to(self._dummy.device, non_blocking=True))
            allFeatures.append(x.cpu())
            total += len(x)
            if progress is not None:
                progress.update(task, advance=1, progress=f"{total:4d} queries")
        # [N, D]
        allFeatures = torch.cat(allFeatures)
        return torch.from_numpy(self.database.search(allFeatures.numpy(), numReturns))
