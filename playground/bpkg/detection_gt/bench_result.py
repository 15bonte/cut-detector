from dataclasses import dataclass

@dataclass
class BenchResult:
    min_dist: float
    max_dist: float
    avg_dist: float

    n_miss: int

    same_method_bench_gt: bool