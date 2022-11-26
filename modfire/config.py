from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional

from marshmallow import Schema, fields, post_load, RAISE


class GeneralSchema(Schema):
    class Meta:
        unknown = RAISE
    key = fields.Str(required=True, description="A unique key used to retrieve in registry. For example, given `Lamb` for optimizers, it will check `OptimRegistry` and find the optimizer `apex.optim.FusedLAMB`.")
    params = fields.Dict(required=True, description="Corresponding funcation call parameters. So the whole call is `registry.get(key)(**params)`.")

    @post_load
    def _(self, data, **kwargs):
        return General(**data)

# Ununsed, unless you want ddp training
class GPUSchema(Schema):
    class Meta:
        unknown = RAISE
    gpus = fields.Int(required=True, description="Number of gpus for training. This affects the `world size` of PyTorch DDP.", exclusiveMinimum=0)
    vRam = fields.Int(required=True, description="Minimum VRam required for each gpu. Set it to `-1` to use all gpus.")
    wantsMore = fields.Bool(required=True, description="Set to `true` to use all visible gpus and all VRams and ignore `gpus` and `vRam`.")

    @post_load
    def _(self, data, **kwargs):
        return GPU(**data)

class TrainSchema(Schema):
    class Meta:
        unknown = RAISE
    batchSize = fields.Int(required=True, description="Batch size for training. NOTE: The actual batch size (whole world) is computed by `batchSize * gpus`.", exclusiveMinimum=0)
    epoch = fields.Int(required=True, description="Total training epochs.", exclusiveMinimum=0)
    valFreq = fields.Int(required=True, description="Run validation after every `valFreq` epochs.", exclusiveMinimum=0)
    trainSet = fields.Str(required=True, description="A txt path to load images per line.")
    searchSet = fields.Str(required=True, description="A txt path to load images per line for database.")
    querySet = fields.Str(required=True, description="A txt path to load images per line for query.")
    saveDir = fields.Str(required=True, description="A dir path to save model checkpoints, TensorBoard messages and logs.")
    ckptPath = fields.Str(required=True, description="Path to restore model ckpt for warm training.")
    debug = fields.Bool(required=True, description="Debug mode flag.")
    optim = fields.Nested(GeneralSchema(), required=True, description="Optimizer used for training. As for current we have `Adam` and `Lamb`.")
    schdr = fields.Nested(GeneralSchema(), required=True, description="Learning rate scheduler used for training. As for current we have `ReduceLROnPlateau`, `Exponential`, `MultiStep`, `OneCycle` and all schedulers defined in `mcquic.train.lrSchedulers`.")
    gpu = fields.Nested(GPUSchema(), required=True, description="GPU configs for training.")
    hooks = fields.List(fields.Nested(GeneralSchema()), required=False, description="Hooks used for training. Key is used to retrieve hook from `LBHash.train.hooks`.")
    externalLib = fields.List(fields.Str(), required=False, allow_none=True, description="External libraries used for training. All python files in `externalLib` will be imported as modules. In this way, you could extend registries.")

    @post_load
    def _(self, data, **kwargs):
        return Train(**data)

class ConfigSchema(Schema):
    class Meta:
        unknown = RAISE
    model = fields.Nested(GeneralSchema(), required=True, description="Hashing model to use. Avaliable params are `backbone`, `bits` and `hashMethod`.")
    train = fields.Nested(TrainSchema(), required=True, description="Training configs.")

    @post_load
    def _(self, data, **kwargs):
        return Config(**data)



@dataclass
class General:
    key: str
    params: Dict[str, Any]

    @property
    def Key(self) -> str:
        return self.key

    @property
    def Params(self) -> Dict[str, Any]:
        return self.params

@dataclass
class GPU:
    gpus: int
    vRam: int
    wantsMore: bool

    @property
    def GPUs(self) -> int:
        return self.gpus

    @property
    def VRam(self) -> int:
        return self.vRam

    @property
    def WantsMore(self) -> bool:
        return self.wantsMore

@dataclass
class Train:
    batchSize: int
    epoch: int
    valFreq: int
    trainSet: str
    searchSet: str
    querySet: str
    saveDir: str
    target: str
    optim: General
    schdr: General
    debug: bool
    ckptPath: str
    gpu: GPU
    hooks: Optional[List[General]] = None
    externalLib: Optional[List[str]] = None

    @property
    def BatchSize(self) -> int:
        return self.batchSize

    @property
    def Epoch(self) -> int:
        return self.epoch

    @property
    def ValFreq(self) -> int:
        return self.valFreq

    @property
    def TrainSet(self) -> str:
        return self.trainSet

    @property
    def SearchSet(self) -> str:
        return self.searchSet

    @property
    def QuerySet(self) -> str:
        return self.querySet

    @property
    def SaveDir(self) -> str:
        return self.saveDir

    @property
    def Target(self) -> str:
        return self.target

    @property
    def Debug(self) -> bool:
        return self.debug

    @property
    def CkptPath(self) -> str:
        return self.ckptPath

    @property
    def Optim(self) -> General:
        if "lr" in self.optim.Params:
            batchSize = self.BatchSize
            exponent = math.log2(batchSize)
            scale = 3 - exponent / 2
            optim = deepcopy(self.optim)
            optim.Params["lr"] /= (2 ** scale)
            return optim
        return self.optim

    @property
    def Schdr(self) -> General:
        return self.schdr

    @property
    def GPU(self) -> GPU:
        return self.gpu

    @property
    def Hooks(self) -> List[General]:
        if self.hooks is None:
            return list()
        return self.hooks

    @property
    def ExternalLib(self) -> List[str]:
        if self.externalLib is None:
            return list()
        return self.externalLib


@dataclass
class Config:
    model: General
    train: Train

    @property
    def Model(self) -> General:
        return self.model

    @property
    def Train(self) -> Train:
        return self.train

    def serialize(self) -> dict:
        return ConfigSchema().dump(self)

    @staticmethod
    def deserialize(data: dict) -> "Config":
        data = { key: value for key, value in data.items() if "$" not in key }
        return ConfigSchema().load(data)
