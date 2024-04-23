import numpy as np

from cnn_framework.utils.readers.tiff_reader import TiffReader

def load_movie(dir: str, filename: str) -> np.array:
    image_path = gen_filepath(dir, filename)
    image = TiffReader(image_path, respect_initial_type=True).image  # TCZYX

    mitosis_movie = image[:, :3, ...].squeeze()  # T C=3 YX
    mitosis_movie = mitosis_movie.transpose(0, 2, 3, 1)  # TYXC

    mask_movie = image[:, 3, ...].squeeze()  # TYX
    mask_movie = mask_movie.transpose(0, 1, 2)  # TYX

    return mitosis_movie


def gen_filepath(dir: str, filename: str) -> str:
    if dir[-1] == "/" or dir[-1] == "\\":
        return f"{dir}{filename}"
    else:
        return f"{dir}/{filename}"

