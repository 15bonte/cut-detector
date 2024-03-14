""" LoG sandbox is a temporary file to try to
implement LoG on midbodies frames t28 to t30 
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.morphology import area_closing, area_opening
from skimage.feature import blob_dog, blob_log, blob_doh
from cnn_framework.utils.readers.tiff_reader import TiffReader
from cut_detector.data.tools import get_data_path
from fake_blob_log import fake_blob_log

def mid_body_log_sandbox(image_path):
    # If image_path is a directory, take its first file
    if os.path.isdir(image_path):
        image_path = os.path.join(image_path, os.listdir(image_path)[0])

    # Read image
    image = TiffReader(image_path, respect_initial_type=True).image  # TCZYX
    mitosis_movie = image[:, :3, ...].squeeze()  # T C=3 YX
    mitosis_movie = mitosis_movie.transpose(0, 2, 3, 1)  # TYXC
    mask_movie = image[:, 3, ...].squeeze()  # TYX
    mask_movie = mask_movie.transpose(0, 1, 2)  # TYX

    detect_movie_spots(mitosis_movie)

def detect_movie_spots(movie):
    mid_body_chan = 1
    sir_chan = 0
    nb_frames = movie.shape[0]

    for frame in range(nb_frames):
        print(f"processing frame {frame}:")
        mitosis_frame = movie[frame, :, :, :].squeeze()  # YXC
        # process_frame_bin(mitosis_frame, mid_body_chan, sir_chan, frame)
        process_frame_nonbin(mitosis_frame, mid_body_chan, sir_chan, frame)
        # input("press enter to process next frame")
    plt.show()

def process_frame_bin(image: np.array, mb_chan: int, sir_chan: int, frame_n: int):
    image_sir = image[:, :, sir_chan]
    image_mklp = image[:, :, mb_chan]
    # 248 260
    # if frame_n == 0: print(image_mklp[255:265, 245:250])
    # 268 254
    # if frame_n == 1: print(image_mklp[249:259, 263:273])
    # 268 180
    # if frame_n == 2: print(image_mklp[175:185, 263:273])
    # imshow(image_mklp)

    ## Threshold on image
    offset = 125
    # offset = 135
    binary_image = image_mklp > offset
    
    ## Area closing and opening
    binary_image = area_closing(area_opening(binary_image, 50), 200)

    ## Blob Log
    # blobs = blob_log(binary_image, min_sigma=3, max_sigma=6, num_sigma=30, threshold=.1)
    blobs = fake_blob_log(binary_image, min_sigma=3, max_sigma=6, num_sigma=30, threshold=.1, plot_cube=True)
    # blobs[:, 2] = blobs[:, 2] * np.sqrt(2)
    print(blobs)

    ## output
    plt.figure()
    plt.imshow(binary_image)


def process_frame_nonbin(image: np.array, mb_chan: int, sir_chan: int, frame_n: int):
    image_sir = image[:, :, sir_chan]
    image_mklp = image[:, :, mb_chan]
    raw_image = image_mklp / np.max(image_mklp) # Maybe really important ?

    ## Area closing and opening
    # raw_image = area_closing(area_opening(raw_image, 25), 25)

    ## Blob Log
    blobs = []
    blobs = fake_blob_log(raw_image, min_sigma=5, max_sigma=10, num_sigma=30, threshold=.1, plot_cube=True)
    # blobs = blob_log(raw_image, min_sigma=5, max_sigma=10, num_sigma=30, threshold=.1)
    # blobs[:, 2] = blobs[:, 2] * np.sqrt(2)
    print("blobs:", blobs)
    print("blobs len:", blobs.shape[0])

    ## output
    # plt.figure()
    # plt.imshow(raw_image)

if __name__ == "__main__":
    mid_body_log_sandbox(
        "./src/cut_detector/data/mitosis_movies/example_video_mitosis_t28-30.tiff"
    )
