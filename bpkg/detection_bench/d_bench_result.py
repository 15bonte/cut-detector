from typing import Dict

from detection_gt import BenchStat

class DBenchResult:
    dist_results: Dict[str, BenchStat]
    time_results: Dict[str, float]
    
    def __init__(self):
        self.dist_results = {}
        self.time_results = {}