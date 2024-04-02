from dataclasses import dataclass

@dataclass
class BenchStat:
    min_dist: float
    max_dist: float
    avg_dist: float

    n_miss: int

    same_method_bench_gt: bool