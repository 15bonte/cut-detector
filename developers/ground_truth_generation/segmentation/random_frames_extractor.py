import glob
import os
from pathlib import Path
import numpy as np

from cnn_framework.utils.readers.tiff_reader import TiffReader
from cnn_framework.utils.display_tools import display_progress
from cnn_framework.utils.enum import ProjectMethods
from cnn_framework.utils.readers.utils.projection import Projection
from cnn_framework.utils.tools import save_tiff


def extract_frames(
    save_dir: str,
    data_dir: str,
    time_dimension: str,
    frames_per_file: str,
    projection: list[Projection],
) -> None:
    """Extract random frames from tiff files in a folder.

    Parameters
    ----------
    save_dir : str
        Path to the folder where to save the images.
    data_dir : str
        Path to the folder containing the tiff files.
    time_dimension : int
        Time dimension in the tiff file.
    frames_per_file : int
        How many frames to extract from each video.
    projection : list[Projection]
        List of projections to apply to the image.

    """
    # Create save directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Select tif or tiff files in folder
    files = glob.glob(os.path.join(data_dir, "*.tif*"))
    # Iterate over files
    for file_index, file in enumerate(files):
        # Read image
        file_path = os.path.join(data_dir, file)
        image = TiffReader(
            file_path, project=projection, respect_initial_type=True
        ).get_processed_image()
        # Select random time steps to extract
        time_points_to_extract = np.random.choice(
            image.shape[time_dimension] - 1, frames_per_file, replace=False
        )
        time_points_to_extract = np.unique(time_points_to_extract).astype(int)
        # Iterate over time stamps
        selected_images = image.take(
            time_points_to_extract, axis=time_dimension
        )
        # Save frames
        for frame, selected_image in zip(
            time_points_to_extract, selected_images
        ):
            padded_frame = str(frame).zfill(
                3
            )  # so that cellpose take them in order
            selected_image = np.expand_dims(selected_image, axis=0)
            save_tiff(
                selected_image,
                f"{save_dir}/{padded_frame}_{Path(file).stem}.tif",
                original_order="TZCYX",
            )

        display_progress(
            "Generation in progress",
            file_index + 1,
            len(files),
        )


if __name__ == "__main__":
    DATA_DIR = ""
    SAVE_DIR = ""

    FRAMES_PER_FILE = 10  # how many frames to extract from each video
    TIME_DIMENSION = 0  # time dimension in the tiff file

    PROJECTION = [
        Projection(method=ProjectMethods.Channel, channels=[2], axis=1)
    ]  # channel 2 on dimension 1

    extract_frames(
        SAVE_DIR, DATA_DIR, TIME_DIMENSION, FRAMES_PER_FILE, PROJECTION
    )
