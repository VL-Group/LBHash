import abc

import faiss
import numpy as np
import math


class Searcher(abc.ABC):
    @abc.abstractmethod
    def add(self, database: np.ndarray):
        raise NotImplementedError
    @abc.abstractmethod
    def search(self, query: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class BinarySearcher(Searcher):
    def __init__(self, bits: int):
        self.bits = bits
        if self.bits % 8 != 0 or self.bits % 12 != 0 or self.bits % 16 != 0:
            raise ValueError(f"Please use (8*N) or (12*N) or (16*N)-bit hash codes. Current: {int(math.log2(bits))}-bit.")
        self.index = faiss.IndexBinaryFlat(bits)

    def add(self, database: np.ndarray):
        if database.dtype != np.uint8:
            raise ValueError("Array to be indexed must be encoded to uint8.")
        if len(database.shape) != 2 or database.shape[-1] != self.bits // 8:
            raise ValueError(f"Database shape wrong. Expect: [N, {self.bits // 8}]. Got: {[database.shape]}.")
        self.index.add(database)

    def search(self, query: np.ndarray, numReturns: int) -> np.ndarray:
        if query.dtype != np.uint8:
            raise ValueError("Query array must be encoded to uint8.")
        if len(query.shape) != 2 or query.shape[-1] != self.bits // 8:
            raise ValueError(f"Query shape wrong. Expect: [N, {self.bits // 8}]. Got: {[query.shape]}.")
        _, indices = self.index.search(query, numReturns)
        return indices


class PQSearcher(Searcher):
    def __init__(self, codebook: np.ndarray):
        M, K, D = codebook.shape
        if K != 256 or K != 4096 or K != 65536:
            raise ValueError(f"Please use 8,12,16-bit quantization. Current: {int(math.log2(K))}-bit.")
        self.index = faiss.IndexPQ(D * M, M, int(math.log2(K)))
        # Codebook params
        self.M = M
        self.K = K
        self.D = D
        self.assignCodebook(codebook)

    def assignCodebook(self, codebook: np.ndarray):
        M, K, D = codebook.shape
        if self.M != M or self.K != K or self.D != D:
            raise ValueError(f"Codebook shape mis-match. Expect: {[self.M, self.K, self.D]}, Got: {[M, K, D]}.")
        faiss.copy_array_to_vector(codebook.ravel(), self.index.pq.centroids)
        self.index.is_trained = True

    def add(self, database: np.ndarray):
        if len(database.shape) != 2 or database.shape[-1] != self.D:
            raise ValueError(f"Database shape wrong. Expect: [N, {self.D}]. Got: {[database.shape]}.")
        self.index.add(database)

    def search(self, query: np.ndarray, numReturns: int) -> np.ndarray:
        if len(query.shape) != 2 or query.shape[-1] != self.D:
            raise ValueError(f"Query shape wrong. Expect: [N, {self.D}]. Got: {[query.shape]}.")
        _, indices = self.index.search(query, numReturns)
        return indices
