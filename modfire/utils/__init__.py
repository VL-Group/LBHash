from ._api import findLastLinear, replaceModule

import logging
import os
import signal
from typing import List, Optional, Union, Any, Dict, Tuple
from pathlib import Path
import hashlib
from time import sleep
import threading
import abc

from vlutils.logger import LoggerBase
from vlutils.runtime import functionFullName
from vlutils.custom import RichProgress
from vlutils.saver import Saver, DummySaver, StrPath
from vlutils.base import FrequecyHook
import torch
from torch import nn
from rich import filesize
from rich.progress import Progress
from rich.progress import TimeElapsedColumn, BarColumn, TimeRemainingColumn

from modfire import Consts


def nop(*_, **__):
    pass


def totalParameters(model: nn.Module) -> str:
    allParams = sum(p.numel() for p in model.parameters())
    unit, suffix = filesize.pick_unit_and_suffix(allParams, ["", "k", "M", "B"], 1000)
    return f"{(allParams / unit):.4f}{suffix}"


def hashOfFile(path: StrPath, progress: Optional[Progress] = None):
    sha256 = hashlib.sha256()

    fileSize = os.path.getsize(path)

    if progress is not None:
        task = progress.add_task(f"[ Hash ]", total=fileSize, progress="0.00%", suffix="")

    now = 0

    with open(path, 'rb') as fp:
        while True:
            # Reading is buffered, so we can read smaller chunks.
            chunk = fp.read(65536)
            if not chunk:
                break
            sha256.update(chunk)
            now += 65536
            if progress is not None:
                progress.update(task, advance=65536, progress=f"{now / fileSize * 100 :.2f}%")

    if progress is not None:
        progress.remove_task(task)

    hashResult = sha256.hexdigest()
    return hashResult

def checkHook(function, name, logger: Union[logging.Logger, LoggerBase]=logging.root):
    if function is None:
        logger.debug("No <%s>.", name)
        return nop
    fullName = functionFullName(function)
    logger.debug("<%s> is `%s`.", name, fullName)
    return function


def getRichProgress(disable: bool = False) -> RichProgress:
    return RichProgress("[i blue]{task.description}[/][b magenta]{task.fields[progress]}", TimeElapsedColumn(), BarColumn(None), TimeRemainingColumn(), "{task.fields[suffix]}", refresh_per_second=6, transient=True, disable=disable, expand=True)


def getSaver(saveDir: StrPath, saveName: StrPath = "saved.ckpt", loggerName: str = "root", loggingLevel: Union[str, int] = "INFO", config: Any = None, autoManage: bool = True, maxItems: int = 25, reserve: bool = False, dumpFile: Optional[str] = None, activateTensorboard: bool = True, disable: bool = False):
    if disable:
        return DummySaver(saveDir, saveName, loggerName, loggingLevel, config, autoManage, maxItems, reserve, dumpFile, activateTensorboard)
    else:
        return Saver(saveDir, saveName, loggerName, loggingLevel, config, autoManage, maxItems, reserve, dumpFile, activateTensorboard)

getSaver.__doc__ = Saver.__doc__

class EpochFrequencyHook(FrequecyHook):
    def __call__(self, step: int, epoch: int, *args: Any, **kwArgs: Any) -> Dict[int, Any]:
        with torch.inference_mode():
            return super().__call__(epoch, step, epoch, *args, **kwArgs)

class EMATracker(nn.Module):
    def __init__(self, size: Union[torch.Size, List[int], Tuple[int, ...]], momentum: float = 0.9):
        super().__init__()
        self._shadow: torch.Tensor
        self._decay = 1 - momentum
        self.register_buffer("_shadow", torch.empty(size) * torch.nan)

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        if torch.all(torch.isnan(self._shadow)):
            self._shadow.copy_(x)
            return self._shadow
        self._shadow -= self._decay * (self._shadow - x)
        return self._shadow



@staticmethod
def formatStep(step):
    unit, suffix = filesize.pick_unit_and_suffix(step, ["", "k", "M"], 1000)
    if unit < 10:
        return f"{(step // unit):5d}"
    else:
        truncated = step / unit
        if truncated < 10:
            return f"{truncated:4.6f}"[:4] + suffix
        elif truncated < 100:
            return f"{truncated:4.6f}"[:4] + suffix
        else:
            return f"{truncated:11.6f}"[:4] + suffix


class PrettyStep:
    def __init__(self):
        self._lastFormatted = -1
        self._prettyStep = "......"

    def __call__(self, step) -> str:
        if step == self.lastFormatted:
            return self.prettyStep
        else:
            self.prettyStep = formatStep(step)
            self.lastFormatted = step
            return self.prettyStep


class SafeTerminate(abc.ABC):
    def __init__(self, logger: Optional[LoggerBase]= None) -> None:
        self.logger = logger or logging
        signal.signal(signal.SIGTERM, self._terminatedHandler)

    def _kill(self):
        sleep(Consts.TimeOut)
        self.logger.critical("Timeout exceeds, killed.")
        signal.raise_signal(signal.SIGKILL)

    # Handle SIGTERM when main process is terminated.
    # Save necessary info.
    def _terminatedHandler(self, signum, frame):
        killer = threading.Thread(target=self._kill, daemon=True)
        killer.start()
        self.onTerminate(signum, frame)

        self.logger.critical("[%s] QUIT.")
        # reset to default SIGTERM handler
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.raise_signal(signal.SIGTERM)


    @abc.abstractmethod
    def onTerminate(self, signum, frame):
        raise NotImplementedError
