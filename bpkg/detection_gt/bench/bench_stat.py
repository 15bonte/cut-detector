import numpy as np
from dataclasses import dataclass

@dataclass
class BenchStat:
    distances: np.ndarray
    n_miss: int

    same_method_bench_gt: bool

    def min(self) -> float:
        return np.min(self.distances)

    def max(self) -> float:
        return np.max(self.distances)

    def avg(self) -> float:
        return np.average(self.distances)

    def median(self) -> float:
        return np.median(self.distances)

    def percentile(self, p: int) -> float:
        return np.percentile(self.distances, q=p)