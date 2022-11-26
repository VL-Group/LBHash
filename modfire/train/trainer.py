import functools
import os
import shutil
from typing import Callable, Optional, Tuple, Type, Any
import gc
import pathlib
import hashlib
import importlib.util
import sys

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch import distributed as dist
from vlutils.base import Registry
from vlutils.base.freqHook import ChainHook
from vlutils.saver import Saver
from vlutils.logger import trackingFunctionCalls
from vlutils.base import Restorable
from vlutils.runtime import relativePath

import modfire.utils.registry
from modfire import Consts
from modfire.config import Config
from modfire.train.hooks import getAllHooks
from modfire.validate import Validator
from modfire.utils import totalParameters, StrPath, checkHook, getRichProgress, EpochFrequencyHook, EMATracker, PrettyStep, SafeTerminate, getSaver
from modfire.utils.registry import OptimRegistry, SchdrRegistry, CriterionRegistry, ModelRegistry, FunctionRegistry

class PalTrainer(Restorable):
    def __init__(self, config: Config, loggingLevel: int):
        super().__init__()
        self.rank = dist.get_rank()
        self.worldSize = dist.get_world_size()
        torch.cuda.set_device(self.rank)
        self.config = config
        self.saver = getSaver(self.config.Train.SaveDir, saveName="saved.ckpt", config=config.serialize(), loggerName=Consts.Name, reserve=False, loggingLevel=loggingLevel, disable=self.rank != 0)
        prettyStep = PrettyStep()
        self.saver.decorate(lambda: prettyStep(self._step))

        self.saver.debug("<%s> is located at rank `%d`", self.__class__.__name__, self.rank)

        self._epoch = 0
        self._step = 0

        # # Used for self.PrettyStep
        # self.lastFormatted = -1
        self._preRegistration(config, self.saver)

        self._model, self.modelFn = self._createModel(self.rank, self.config, self.saver)
        self._criterion, self.criterionFn = self._createCriterion(self.rank, self.config, self.saver)
        self._optimizer, self.optimFn = self._createOptimizer(self.config, self._model, self.saver)
        self._scheduler, self.schdrFn = self._createScheduler(self.config, self._optimizer, self.saver)

        self.saver.debug("<%s> created.", self.__class__.__name__)

    def save(self, path = None):
        self.saver.save(path, trainer=self, config=self.config.serialize())

    def restoreStates(self, path: StrPath):
        self.saver.debug("Restored state dict from `%s`", path)
        self.saver.load(path, "cpu", logger=self.saver, trainer=self)
        self.saver.debug("Restore network parameters finished.")
        self.resetOptimizer()
        self.resetScheduler(self._scheduler.last_epoch)

    def resetOptimizer(self):
        del self._optimizer
        self._optimizer = self.optimFn(self._model.parameters(), **self.config.Train.Optim.Params)

        for group in self._optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])

        self.saver.debug("Optimizer reset.")

    def resetScheduler(self, lastEpoch=-1):
        del self._scheduler
        self._scheduler = self.schdrFn(self._optimizer, last_epoch=lastEpoch, **self.config.Train.Schdr.Params)
        self.saver.debug("LR scheduler reset.")

    def train(self):
        beforeRunHook, afterRunHook, stepStartHook, stepFinishHook, epochStartHook, epochFinishHook = self._createHooks(self.config, self.saver)

        trainLoader, searchLoader, queryLoader = self._createDataLoaders(self.config)

        trainingArgs = {
            "trainLoader": trainLoader,
            "searchLoader": searchLoader,
            "queryLoader": queryLoader
        }

        self._beforeRun(beforeRunHook, **trainingArgs)

        for i in range(self._epoch, self.config.Train.Epoch):
            self._epochStart(epochStartHook, **trainingArgs)
            trainLoader.sampler.set_epoch(i)
            self._runAnEpoch(stepStartHook, stepFinishHook, trainLoader, trainingArgs)
            self._epochFinish(epochFinishHook, trainSet=trainLoader.dataset, **trainingArgs)
        self._afterRun(afterRunHook)

    def _runAnEpoch(self, stepStartHook, stepFinishHook, trainLoader, trainingArgs):
        self._model.train()
        for images, targets in trainLoader:
            images = images.cuda(non_blocking=True)

            self._stepStart(stepStartHook, **trainingArgs)

            self._optimizer.zero_grad()
            z = self._model(images)
            loss = self._criterion(z, targets)
            loss.backward()
            self._optimizer.step()

            self._stepFinish(stepFinishHook, loss=loss, **trainingArgs)

    @staticmethod
    def _preRegistration(config: Config, saver: Saver):
        otherPythonFiles = config.Train.ExternalLib
        for pyFile in otherPythonFiles:
            filePath = pathlib.Path(pyFile).absolute()
            # md5 of abs file path as module name
            moduleName = hashlib.md5(str(filePath).encode()).hexdigest()
            spec = importlib.util.spec_from_file_location(moduleName, pyFile)
            if spec is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[moduleName] = module
            spec.loader.exec_module(module)

        for reg in modfire.utils.registry.__all__:
            registry = getattr(modfire.utils.registry, reg)
            if issubclass(registry, Registry):
                saver.debug("Summary of %s: \r\n%s", registry, registry.summary())

    @staticmethod
    def _createHooks(config: Config, saver: Saver):
        allHooks = getAllHooks(config.Train.Hooks)
        beforeRunHook, afterRunHook, stepStartHook, stepFinishHook, epochStartHook, epochFinishHook = allHooks["beforeRunHook"], allHooks["afterRunHook"], allHooks["stepStartHook"], allHooks["stepFinishHook"], allHooks["epochStartHook"], allHooks["epochFinishHook"]
        beforeRunHook = checkHook(beforeRunHook, "BeforeRunHook", saver)
        afterRunHook = checkHook(afterRunHook, "AfterRunHook", saver)
        stepStartHook = checkHook(stepStartHook, "StepStartHook", saver)
        stepFinishHook = checkHook(stepFinishHook, "StepFinishHook", saver)
        epochStartHook = checkHook(epochStartHook, "EpochStartHook", saver)
        epochFinishHook = checkHook(epochFinishHook, "EpochFinishHook", saver)
        return beforeRunHook, afterRunHook, stepStartHook, stepFinishHook, epochStartHook, epochFinishHook

    @staticmethod
    def _createDataLoaders(config: Config, saver: Saver):
        saver.debug("Train and validation datasets mounted.")

    @staticmethod
    def _createModel(rank: int, config: Config, saver: Saver) -> Tuple[DistributedDataParallel, Callable[..., nn.Module]]:
        saver.debug("Creating model...")
        modelFn = trackingFunctionCalls(ModelRegistry.get(config.Model.Key), saver)
        model = modelFn(**config.Model.Params)
        model = DistributedDataParallel(model.to(rank), device_ids=[rank], output_device=rank, find_unused_parameters=False)
        saver.debug("Model created. Size: %s.", totalParameters(model))
        return model, modelFn

    @staticmethod
    def _createOptimizer(config: Config, model: DistributedDataParallel, saver: Saver) -> Tuple[torch.optim.Optimizer, Callable[..., torch.optim.Optimizer]]:
        saver.debug("Creating optimizer...")
        optimFn = trackingFunctionCalls(OptimRegistry.get(config.Train.Optim.Key), saver)
        optimizer = optimFn(model.parameters(), **config.Train.Optim.Params)
        saver.debug("Optimizer created.")
        return optimizer, optimFn

    @staticmethod
    def _createScheduler(config: Config, optimizer: torch.optim.Optimizer, saver: Saver) -> Tuple[torch.optim.lr_scheduler._LRScheduler, Callable[..., torch.optim.lr_scheduler._LRScheduler]]:
        saver.debug("Creating LR scheduler...")
        schdrFn = trackingFunctionCalls(SchdrRegistry.get(config.Train.Schdr.Key), saver)
        scheduler = schdrFn(optimizer, **config.Train.Schdr.Params)
        saver.debug("LR scheduler created.")
        return scheduler, schdrFn

    @staticmethod
    def _createCriterion(rank: int, config: Config, saver: Saver) -> Tuple[nn.Module, Callable[..., nn.Module]]:
        saver.debug("Creating criterion...")
        criterionFn = trackingFunctionCalls(CriterionRegistry.get(config.Train.Criterion.Key), saver)
        criterion = criterionFn(**config.Train.Criterion.Params).to(rank)
        saver.debug("criterion created.")
        return criterion, criterionFn

    def _beforeRun(self, hook, *args, **kwArgs):
        if self.config.Train.CkptPath is not None:
            self.restoreStates(self.config.Train.CkptPath)
            self.saver.info("Resume training at %d epochs.", self._epoch)
        self.saver.info("Start training.")

        hook(self._step, self._epoch, self, *args, logger=self.saver, **kwArgs)

        self.saver.info("See you at `%s`", self.saver.TensorboardURL)

    def _afterRun(self, hook, *args, **kwArgs):
        self.saver.debug("Training loop finished.")
        hook(self._step, self._epoch, self, *args, logger=self.saver, **kwArgs)

    def _stepStart(self, hook, *args, **kwArgs):
        hook(self._step, self._epoch, self, *args, logger=self.saver, **kwArgs)

    def _stepFinish(self, hook, *args, loss, **kwArgs):
        self._step += 1
        hook(self._step, self._epoch, self, *args, logger=self.saver, loss=loss, **kwArgs)

    def _epochStart(self, hook, *args, trainLoader, **kwArgs):
        self.saver.debug("Epoch %4d started.", self._epoch + 1)

        gc.collect()
        gc.collect()
        hook(self._step, self._epoch, self, *args, trainLoader=trainLoader, logger=self.saver, **kwArgs)

    def _epochFinish(self, hook, *args, **kwArgs):
        self._epoch += 1

        self.saver.debug("Epoch %4d finished.", self._epoch)

        self._scheduler.step()
        self.saver.debug("Lr is set to %.2e.", self._scheduler.get_last_lr()[0])

        hook(self._step, self._epoch, self, *args, logger=self.saver, **kwArgs)


class MainTrainer(PalTrainer, SafeTerminate):
    def __init__(self, config: Config, loggingLevel: int):
        # Running depedencies
        self.progress = getRichProgress().__enter__()
        self.trainingBar = self.progress.add_task("", start=False, progress="[----/----]", suffix=Consts.CDot * 10)
        self.epochBar = self.progress.add_task("[----/----]", start=False, progress="", suffix=Consts.CDot * 10)

        self.validator = Validator(self.config)

        self.diffTracker = EMATracker((), 0.99).cuda()

        PalTrainer.__init__(self, config, loggingLevel)
        SafeTerminate.__init__(self, self.saver)
        # Logging and saving
        self.bestmAP = -1
        # Call function at every X epoches.
        self.epochFinishCalls = EpochFrequencyHook(
            (1, self.log),
            logger=self.saver
        )
        self.epochStartCalls = EpochFrequencyHook(
            (self.config.Train.ValFreq, self.validate),
            logger=self.saver
        )

    def onTerminate(self, signum, frame):
        self.saver.critical("Main process was interrupted, try to save necessary info.")
        self.saver.critical("This post-process will be killed after %d secs if stuck.", Consts.TimeOut)
        self.progress.__exit__(None, None, None)
        self.save(os.path.join(self.saver.SaveDir, "last.ckpt"))
        self.saver.critical("Find the last checkpoint at `%s`", relativePath(os.path.join(self.saver.SaveDir, "last.ckpt")))
        self.summary()
        self.saver.critical("QUIT.")

    def summary(self):
        if self.bestmAP < 0:
            self.saver.info("Total epoches: %d, total steps: %s, best mAP: N/A.", self._epoch, self._step)
        else:
            self.saver.info("Total epoches: %d, total steps: %s, best mAP: %.2f%%.", self._epoch, self._step, self.bestmAP * 100)
        self.saver.info("Model saved to %s`.", relativePath(os.path.join(self.saver.SaveDir, "[ONE_OF_A].ckpt")))

    def _beforeRun(self, hook, *args, **kwArgs):
        self.progress.start_task(self.trainingBar)
        self.progress.start_task(self.epochBar)
        super()._beforeRun(hook, *args, **kwArgs)

    def _afterRun(self, hook, *args, **kwArgs):
        self.progress.__exit__(None, None, None)
        super()._afterRun(hook, *args, **kwArgs)
        self.summary()

    def _stepFinish(self, hook, *args, loss, **kwArgs):
        super()._stepFinish(hook, *args, loss=loss, **kwArgs)

        moment = self.diffTracker(loss)

        task = self.progress.get_task(self.trainingBar)
        self.progress.update(self.trainingBar, advance=1, progress=f"[{task.completed + 1:4d}/{task.total:4d}]", suffix=f"L = [b green]{moment:2.2f}[/]")
        self.progress.update(self.epochBar, advance=1)

        if self._step % 100 != 0:
            return
        self.saver.add_scalar(f"Stat/{self.config.Train.Target}", moment, global_step=self._step)
        self.saver.add_scalar(f"Stat/Loss", loss, global_step=self._step)
        self.saver.add_scalar("Stat/Lr", self._scheduler.get_last_lr()[0], global_step=self._step)

    def _epochStart(self, hook, *args, trainLoader, **kwArgs):
        totalBatches = len(trainLoader)
        self.progress.update(self.trainingBar, total=totalBatches)
        self.progress.update(self.epochBar, total=self.config.Train.Epoch * totalBatches, completed=self._step, description=f"[{self._epoch + 1:4d}/{self.config.Train.Epoch:4d}]")

        self.progress.reset(self.trainingBar)
        super()._epochStart(hook, *args, trainLoader=trainLoader, **kwArgs)

    def _createHooks(self, config: Config, saver: Saver):
        allHooks = getAllHooks(config.Train.Hooks)
        beforeRunHook, afterRunHook, stepStartHook, stepFinishHook, epochStartHook, epochFinishHook = allHooks["beforeRunHook"], allHooks["afterRunHook"], allHooks["stepStartHook"], allHooks["stepFinishHook"], allHooks["epochStartHook"], allHooks["epochFinishHook"]
        beforeRunHook = checkHook(beforeRunHook, "BeforeRunHook", saver)
        afterRunHook = checkHook(ChainHook(self.validate, afterRunHook), "AfterRunHook", saver)
        stepStartHook = checkHook(stepStartHook, "StepStartHook", saver)
        stepFinishHook = checkHook(stepFinishHook, "StepFinishHook", saver)
        epochStartHook = checkHook(ChainHook(
            EpochFrequencyHook(
                (config.Train.ValFreq, self.validate), logger=saver
            ), epochStartHook), "EpochStartHook", saver)
        epochFinishHook = checkHook(ChainHook(
            EpochFrequencyHook(
                (1, self.log), logger=saver
            ), epochFinishHook), "EpochFinishHook", saver)
        return beforeRunHook, afterRunHook, stepStartHook, stepFinishHook, epochStartHook, epochFinishHook

    def log(self, *_, **__):
        self.saver.add_scalar("Stat/Epoch", self._epoch, self._step)
        # self.saver.add_images("Train/Raw", tensorToImage(images), global_step=self._step)

    def validate(self, *_, queryLoader: DataLoader, searchLoader: DataLoader, **__):
        torch.cuda.empty_cache()

        self.saver.debug("Start validation at epoch %4d.", self._epoch)

        self._model.eval()
        results, summary = self.validator.validate(self._epoch, self._model, queryLoader, searchLoader, self.progress)

        self.saver.add_scalar(f"Eval/mAP@{self.validator.K}", results["mAP"], global_step=self._step)
        self.saver.add_scalar(f"Eval/Recall@{self.validator.K}", results["Recall"], global_step=self._step)
        self.saver.add_scalar(f"Eval/Precision@{self.validator.K}", results["Precision"], global_step=self._step)
        self.saver.add_images(f"Eval/Visualization", results["Visualization"], global_step=self._step)

        self.save()

        mAP = results["mAP"]

        if mAP > self.bestmAP:
            self.bestmAP = mAP
            self.progress.update(self.epochBar, suffix=f"H = [b red]{self.bestmAP * 100:2.2f}[/]%")
            shutil.copy2(self.saver.SavePath, os.path.join(self.saver.SaveDir, "best.ckpt"))
        self.saver.info("%s", summary)
        self._model.train()

        self.saver.debug("End validation at epoch %4d.", self._epoch)



def getTrainer(rank: int, config: Config, loggingLevel: int):
    if rank == 0:
        return MainTrainer(config, loggingLevel)
    return PalTrainer(config, loggingLevel)
