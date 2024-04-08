""" A module that bridges the gap between the real API
and what is used in the project.
"""

import os
from typing import Optional

from cnn_framework.utils.readers.tiff_reader import TiffReader

from .savers import Saver
from .layers import BlobLayer
from .pipeline import run_pipeline


def run_test_environment(
        paths: list[str],
        pipelines: dict,
        savers: list[list[Saver]],
        out: str = "blobs",
        mlkp_chan: int = 1,
        clear_console: bool = True) -> list[list[any]]:

    # https://stackoverflow.com/a/2084628
    if clear_console:
        os.system('cls' if os.name == 'nt' else 'clear')

    env_rv = []

    for path in paths:
        print("")
        print("####################")
        print("File:", path)
        print("####################")
        print("\n")
        pipeline_rv = []
        for name, pipeline in pipelines.items():
            pipeline_rv.append(simple_run(
                pipeline, 
                path, 
                savers,
                out, 
                mlkp_chan, 
                clear_console=False, 
                pipeline_name=name,
            ))
        env_rv.append(pipeline_rv)

    return env_rv


def simple_run(
        pipeline: list[BlobLayer],  
        tiff_path: str, 
        savers: list[Saver],
        out: str = "blobs",
        mlkp_chan: int = 1,
        clear_console: bool = True,
        pipeline_name: Optional[str] = None) -> any:
    
    # https://stackoverflow.com/a/2084628
    if clear_console:
        os.system('cls' if os.name == 'nt' else 'clear')
    
    if pipeline_name:
        print(f"---- Running Pipeline {pipeline_name} ----")
    else:
        print("---- Pipeline start ----")

    # Read movie
    image = TiffReader(tiff_path, respect_initial_type=True).image  # TCZYX
    mitosis_movie = image[:, :3, ...].squeeze()  # T C=3 YX
    mitosis_movie = mitosis_movie.transpose(0, 2, 3, 1)  # TYXC
    # mask_movie = image[:, 3, ...].squeeze()  # TYX
    # mask_movie = mask_movie.transpose(0, 1, 2)  # TYX
    mlkp_movie = mitosis_movie[:,:,:,mlkp_chan]

    return run_pipeline(
        pipeline=pipeline,
        out=out,
        movie=mlkp_movie,
        src_fp=tiff_path,
        logging=True,
        savers=savers,
        pipeline_name=pipeline_name
    )
