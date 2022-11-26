from typing import Any
import abc

import torch
from torch.utils.data import IterDataPipe


class Database(abc.ABC):
    @property
    @abc.abstractmethod
    def DataPipe(self) -> IterDataPipe:
        raise NotImplementedError

    @abc.abstractmethod
    def judge(self, queryInfo: Any, rankList: torch.Tensor) -> torch.Tensor:
        """Return rank list matching result

        Args:
            queryInfo (Any): Information of query, may be indices, labels, etc.
            rankList (torch.Tensor): [len(queries), K] indices, each row represents a rank list of top K from database

        Returns:
            torch.Tensor: [len(queries), K], True or False.
        """
        raise NotImplementedError

class LabeledDatabase(abc.ABC):
    def __init__(self, labels, images):
        super().__init__()
