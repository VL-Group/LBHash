import torch
from torchdata.dataloader2 import DataLoader2
from torch import nn
from rich.progress import Progress
from vlutils.metrics.meter import Meters

from modfire.config import Config
from modfire.model.base import BaseWrapper

class Validator:
    def __init__(self, config: Config):
        self.config = config
        self._meter = Meters(handlers=[
            mAP(config.Train.K).cuda(),
            Precision(config.Train.K).cuda(),
            Recall(config.Train.K).cuda(),
            Visualization().cuda()
        ])

    @torch.no_grad()
    def validate(self, epoch: int, model: BaseWrapper, database: DataLoader2, queryLoader: DataLoader2, progress: Progress):
        model.eval()
        self._meter.reset()
        now = 0

        # Index database
        total = len(searchLoader)
        allLabels = list()

        if epoch is None:
            # test mode
            task = progress.add_task(f"[ Index ]", total=total, progress=f"{now:4d}/{total:4d}", suffix="")
        else:
            task = progress.add_task(f"[ Index@{epoch:4d} ]", total=total, progress=f"{now:4d}/{total:4d}", suffix="")
        for now, (images, targets) in enumerate(searchLoader):
            images = images.cuda(non_blocking=True)
            hashed = model(images)
            progress.update(task, advance=1, progress=f"{(now + 1):4d}/{total:4d}")
        progress.remove_task(task)
        return self._meter.results(), self._meter.summary()
