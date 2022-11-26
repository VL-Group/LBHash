import pathlib
from shutil import copy2
from typing import Union
import os
import random
import logging

import torch
import torch.distributed as dist
from vlutils.config import summary
from vlutils.logger import LoggerBase
import numpy as np

from modfire.config import Config
from modfire import Consts

from .trainer import getTrainer


def initializeBaseConfigs(rank: int, worldSize: int, logger: Union[logging.Logger, LoggerBase] = logging.root):
    # The http socket method fails in some rare cases, we switch to use file
    # os.environ["MASTER_ADDR"] = "127.0.0.1"
    # os.environ["MASTER_PORT"] = port
    # logger.debug("DDP master addr: `%s`", "127.0.0.1")
    # logger.debug("DDP master port: `%s`", port)
    torch.autograd.set_detect_anomaly(False)
    torch.backends.cudnn.benchmark = True

    logger.debug("Autograd detect anomaly = `%s`", False)
    logger.debug("         CuDNN bechmark = `%s`", True)
    torch.manual_seed(3407)
    random.seed(3407)
    np.random.seed(3407)
    logger.debug("            Random seed = `%d`", 3407)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", world_size=worldSize, rank=rank, init_method="file:///tmp/???")
    logger.debug("Process group = `%s`, world size = `%d`", "NCCL", worldSize)

def ddpSpawnTraining(rank: int, worldSize: int, config: Config, resume: pathlib.Path, loggingLevel: int):
    # load ckpt before create trainer, in case it moved to other place.
    if resume is not None:
        if rank == 0:
            tmpFile = copy2(resume, os.path.join(Consts.TempDir, "resume.ckpt"), follow_symlinks=False)
        else:
            tmpFile = os.path.join(Consts.TempDir, "resume.ckpt")
    else:
        tmpFile = None


    logging.info("Here is the whole config during this run: \r\n%s", summary(config.serialize()))

    logging.debug("Creating the world...")
    initializeBaseConfigs(rank, worldSize)
    logging.debug("Base configs initialized.")

    dist.barrier()

    trainer = getTrainer(rank, config, loggingLevel)

    if tmpFile is not None:
        trainer.saver.info("Found ckpt to resume at %s", resume)
        trainer.restoreStates(tmpFile)

    # trainLoader, trainSampler = getTrainLoader(rank, worldSize, config.Train.TrainSet, config.Train.BatchSize, logger=saver)
    # valLoader = getValLoader(config.Train.ValSet, disable=rank != 0, logger=saver)
    # **getAllHooks(config.Train.Hooks)

    trainer.train()

    trainer.saver.debug(summary(config.serialize()))
    trainer.saver.info("Bye.")
