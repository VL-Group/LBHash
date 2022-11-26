import abc
from typing import Iterator, Union, Any

from PIL import Image
import torch
from torch.utils.data import MapDataPipe, IterDataPipe
from torchvision.datasets import CIFAR10 as _c10, CIFAR100 as _c100

from modfire.data import Database

class CIFAR(abc.ABC):
    def __init__(self, root, trainTransform, evalTransform, targetTransform):
        super().__init__()
        allImages, allTargets = self.getAlldata(root, True)
        allLabels = torch.unique(allTargets)
        allTrains, allQueries, allDatabase = list(), list(), list()
        allTrainLabels, allQueryLabels, allDatabaseLabels = list(), list(), list()
        for label in allLabels:
            images = allImages[label == allTargets]
            targets = allTargets[label == allTargets]
            # 1000, 100, 5900 or 100, 10, 590
            trainSize, querySize = images // 6, images // 60
            train, query, database = images[:trainSize], images[trainSize:(trainSize + querySize)], images[(trainSize + querySize):]
            trainLabel, queryLabel, databaseLabel = targets[:trainSize], targets[trainSize:(trainSize + querySize)], targets[(trainSize + querySize):]
            allTrains.append(train)
            allQueries.append(query)
            allDatabase.append(database)
            allTrainLabels.append(trainLabel)
            allQueryLabels.append(queryLabel)
            allDatabaseLabels.append(databaseLabel)
        self.allTrains = torch.cat(allTrains)
        self.allQueries = torch.cat(allQueries)
        self.allDatabase = torch.cat(allDatabase)
        self.allTrainLabels = torch.cat(allTrainLabels)
        self.allQueryLabels = torch.cat(allQueryLabels)
        self.allDatabaseLabels = torch.cat(allDatabaseLabels)
        self.trainTransform = trainTransform
        self.evalTransform = evalTransform
        self.targetTransform = targetTransform

    @staticmethod
    @abc.abstractmethod
    def getAlldata(root, shuffle: bool = True):
        raise NotImplementedError
        # train, test = _c10(root=root, train=True), _c10(root=root, train=False)
        # allImages = np.concatenate((train.data, test.data))
        # allTargets = np.concatenate((np.array(train.targets), np.array(test.targets)))
        # return allImages, allTargets

    @property
    def TrainSplit(self) -> MapDataPipe:
        class _dataPipe(MapDataPipe):
            trains = self.allTrains
            labels = self.allTrainLabels
            transform = self.trainTransform
            target_transform = self.targetTransform
            def __getitem__(self, index):
                img, target = self.data[index], self.targets[index]

                # doing this so that it is consistent with all other datasets
                # to return a PIL Image
                img = Image.fromarray(img)

                if self.transform is not None:
                    img = self.transform(img)

                if self.target_transform is not None:
                    target = self.target_transform(target)
                return img, target

            def __len__(self):
                return len(self.trains)
        return _dataPipe()

    @property
    def QuerySplit(self) -> IterDataPipe:
        class _dataPipe(IterDataPipe):
            queries = self.allQueries
            transform = self.evalTransform
            def __iter__(self) -> Iterator[torch.Tensor]:
                for img in self.queries:
                    # doing this so that it is consistent with all other datasets
                    # to return a PIL Image
                    img = Image.fromarray(img)

                    if self.transform is not None:
                        img = self.transform(img)
                    yield img

            def __len__(self):
                return len(self.database)

        return _dataPipe()

    def _baseSplit(self) -> IterDataPipe:
        class _dataPipe(IterDataPipe):
            database = self.allDatabase
            transform = self.evalTransform
            def __iter__(self) -> Iterator[torch.Tensor]:
                for img in self.database:
                    # doing this so that it is consistent with all other datasets
                    # to return a PIL Image
                    img = Image.fromarray(img)

                    if self.transform is not None:
                        img = self.transform(img)
                    yield img

            def __len__(self):
                return len(self.database)

        return _dataPipe()

    @property
    def Database(self) -> Database:
        class _database(Database):
            _dataPipe = self._baseSplit()
            _queryLabels = self.allQueryLabels
            _baseLabels = self.allDatabaseLabels
            @property
            def DataPipe(self):
                return self._dataPipe
            def judge(self, queryInfo: Any, rankList: torch.Tensor) -> torch.Tensor:
                # NOTE: Here, queryInfo is indices of queries.
                # [Nq]
                queryLabels = self._queryLabels[queryInfo]
                # [Nq, k]
                databaseLabels = self._baseLabels[rankList]
                matching = queryLabels[:, None] == databaseLabels
                return matching
        return _database()
