import pandas as pd
import numpy as np

from pprint import pprint
from time import time
from typing import Callable, Tuple, Any
from skimage.feature import blob_log, blob_doh, blob_dog

class DetectBench:
    def __init__(self, filename: str = "", blog: dict = {}, bdog: dict = {}, bdoh: dict = {}):
        self.df = pd.DataFrame({
            "log_n": [],
            "log_t": [],
            "dog_n": [],
            "dog_t": [],
            "doh_n": [],
            "doh_t": [],
        })

        self.filename = filename
        self.blog_param = blog
        self.bdog_param = bdog
        self.bdoh_param = bdoh

    def bench_img(self, img: np.array):
        t_log, r_log = self.measure_time(blob_log, img, params=self.blog_param)
        t_dog, r_dog = self.measure_time(blob_dog, img, params=self.bdog_param)
        t_doh, r_doh = self.measure_time(blob_doh, img, params=self.bdoh_param)
        self.df.loc[len(self.df)] = {
            "log_n": len(r_log),
            "log_t": t_log,
            "dog_n": len(r_dog),
            "dog_t": t_dog,
            "doh_n": len(r_doh),
            "doh_t": t_doh,
        }

    def measure_time(self, fn: Callable, img: np.array, params: dict) -> Tuple[float, Any]:
        s = time()
        r = fn(image=img, **params)
        e = time()
        return (e-s), r
    
    def print_time(self):
        log_mean, log_median = self.df.loc[:, "log_t"].mean(), self.df.loc[:, "log_t"].median() 
        dog_mean, dog_median = self.df.loc[:, "dog_t"].mean(), self.df.loc[:, "dog_t"].median() 
        doh_mean, doh_median = self.df.loc[:, "doh_t"].mean(), self.df.loc[:, "doh_t"].median()

        dog_median_ratio = dog_median / log_median
        doh_median_ratio = doh_median / log_median

        print("")
        print("=== Time Results ===")
        print("LoG time (avg/median): {:.2f}/{:.2f}ms".format(log_mean*1000, log_median*1000))
        print("DoG time (avg/median): {:.2f}/{:.2f}ms".format(dog_mean*1000, dog_median*1000))
        print("    median ratio: {:.2f}".format(dog_median_ratio))
        print("DoH time (avg/median): {:.2f}/{:.2f}ms".format(doh_mean*1000, doh_median*1000))
        print("    median ratio: {:.2f}".format(doh_median_ratio))

        # print("LoG time (avg/med)(ms):", log_mean*1000, log_median*1000)
        # print("DoG time (avg/med/med-ratio)(ms):", dog_mean*1000, dog_median*1000, dog_median_ratio)
        # print("DoG time (avg/med/med-ratio)(ms):", doh_mean*1000, doh_median*1000, doh_median_ratio)
        print("")

    def print_number_of_found(self):
        log_n = self.df.loc[:, "log_n"].sum()
        dog_n = self.df.loc[:, "dog_n"].sum()
        doh_n = self.df.loc[:, "doh_n"].sum()

        dog_n_ratio = dog_n / log_n
        doh_n_ratio = doh_n / log_n

        print("")
        print("=== Found Stats ===")
        print(f"LoG Found {log_n} blobs")
        print(f"DoG Found {dog_n} blobs")
        print("    {:.2f}".format(dog_n_ratio))
        print(f"DoH Found {doh_n} blobs")
        print("    {:.2f}".format(doh_n_ratio))
        # print("LoG Found", log_n, "blobs")
        # print("DoG Found", dog_n, "blobs (ratio:", dog_n_ratio,")")
        # print("DoH Found", doh_n, "blobs (ratio:", doh_n_ratio,")")
        print("")

    def print_parameters(self):
        print("")
        print("=== Parameters ===")
        print("file:", self.filename)
        print("number of samples:", len(self.df))
        print("")
        print("")
        print("LoG:")
        pprint(self.blog_param)
        print("")
        print("DoG:")
        pprint(self.bdog_param)
        print("")
        print("DoH:")
        pprint(self.bdoh_param)
        print("")

    def print_all(self):
        self.print_parameters()
        self.print_number_of_found()
        self.print_time()
        


