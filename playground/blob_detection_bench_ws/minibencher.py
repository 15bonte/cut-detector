""" Less complex version of the Bencher
"""

import numpy as np

from typing import Callable, Tuple, Dict
from time import time
from skimage.feature import blob_dog, blob_doh, blob_log

class MiniBencher:
    def __init__(
            self, 
            reference: Tuple[str, Tuple[Callable, dict]], 
            candidates: Dict[str, Tuple[Callable, dict]], 
            measure_time: bool = False,
            print_blobs: bool = False):
        
        self.reference      = reference
        self.candidates     = candidates
        
        self.miss = {k: 0 for k in candidates} 
        self.fp   = {k: 0 for k in candidates}
        self.total_time = {k: 0 for k in candidates}
        self.total_time[reference[0]] = 0

        self.ref_blob_count = 0
        self.total_frame  = 0
        self.measure_time = measure_time
        self.print_blobs = print_blobs

    def bench_frame(self, img: np.array):
        self.total_frame += 1
        ref_blobs = self.bench_reference(img)
        self.bench_candidates(img, ref_blobs)

    def bench_reference(self, img: np.array) -> np.array :
        ref_name: str = self.reference[0]
        fn: Callable  = self.reference[1][0]
        args: dict    = self.reference[1][1]

        if self.measure_time:
            start = time()
        blobs = fn(image = img, **args)
        # "log_cur":     (blob_log, {"min_sigma": 5, "max_sigma": 10, "num_sigma":     5, "threshold": 0.1}),
        # blobs = blob_log(image=img, min_sigma=5, max_sigma=10, num_sigma=5, threshold=0.1)
        if self.measure_time: 
             end = time()
             time_taken = end - start
             self.total_time[ref_name] += time_taken

        self.ref_blob_count += len(blobs)
        if self.print_blobs: print(f"{ref_name}: {len(blobs)} blobs: {blobs}")

        return blobs
    
    def bench_candidates(self, image: np.array, ref_blobs: np.array):
        for name, fn_arg in self.candidates.items():
            fn   = fn_arg[0]
            args = fn_arg[1]

            self.bench_candidate(name, fn, args, image, ref_blobs)

    def bench_candidate(self, name: str, fn: Callable, args: dict, image: np.array, ref_blobs: np.array):
        if self.measure_time: 
            start = time()
        blobs = fn(image=image, **args)
        # "dog_cur":     (blob_dog, {"min_sigma": 2, "max_sigma":  5, "sigma_ratio": 1.2, "threshold": 0.1}),
        # blobs = blob_dog(image=image, min_sigma=2, max_sigma=5, sigma_ratio=1.2, threshold=0.1)
        if self.measure_time:
            end = time()
            time_taken = end - start
            self.total_time[name] += time_taken

        if self.print_blobs: print(f"{name}: {len(blobs)} blobs: {blobs}")
        
        delta_blobs_count = len(ref_blobs) - len(blobs)
        if delta_blobs_count < 0:
            self.fp[name] += -delta_blobs_count
        elif delta_blobs_count > 0:
            self.miss[name] += delta_blobs_count

    def print_results(self, image_src: str | None = None):
        print("=============")
        print("")
        print("--- Infos ---")
        if image_src:
            print("image source:", image_src)
        print("")
        print("total frame:", self.total_frame)
        print("Reference function:")
        ref_name = self.reference[0]
        args = self.reference[1][1]
        print(ref_name)
        print(args)
        print("")
        print("Candidates:")
        for name, fn_args in self.candidates.items():
            args = fn_args[1]
            print(f"{name}: {args}")
        print("")
        print("--- Blobs ---")
        print(f"reference found {self.ref_blob_count} blobs")
        print("")
        for cand_name in self.candidates:
            cand_miss = self.miss[cand_name]
            cand_miss_ratio = cand_miss / self.ref_blob_count
            cand_fp = self.fp[cand_name]
            cand_fp_ratio = cand_fp / self.ref_blob_count
            blobs_found = self.ref_blob_count - cand_miss + cand_fp
            blobs_ratio = blobs_found / self.ref_blob_count
            print(f"{cand_name} found {blobs_found}/{self.ref_blob_count} blobs -- {blobs_ratio*100:.2f}% blobs")
            print(f"    {cand_miss}/{self.ref_blob_count} miss | {cand_fp}/{self.ref_blob_count} FP")
            print(f"    {cand_miss_ratio*100:.2f}% miss (m/n_ref) | {cand_fp_ratio*100:.2f}% FP (fp/n_ref)")
        print("")
        if self.measure_time:
            print("--- Time ---")

            ref_time_avg = self.total_time[ref_name] / self.total_frame
            print(f"Reference avg time: {ref_time_avg*1000:.2f}ms")
            print("")
            for cand_name in self.candidates:
                cand_time_avg = self.total_time[cand_name] / self.total_frame
                cand_time_ratio = cand_time_avg / ref_time_avg
                print(f"{cand_name} took {cand_time_avg*1000:.2f}ms (avg)")
                print(f"    x{cand_time_ratio:.2f}")
        print("=============")

            



    




    