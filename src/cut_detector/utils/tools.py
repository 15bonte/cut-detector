import numpy as np


def re_organize_channels(image: np.ndarray) -> np.ndarray:
    """
    Expect a 4 dimensions image.
    Re-organize channels to get TXYC order.
    """
    if image.ndim != 4:
        raise ValueError("Expect a 4 dimensions image.")

    # Get dimension index of smallest dimension, i.e. channels
    channels_dimension_index = np.argmin(image.shape)
    channels_dimension = image.shape[channels_dimension_index]
    if channels_dimension != 3:
        raise ValueError(
            "Expect 3 channels: SiR-tubulin/MKLP1/Phase contrast, in that order."
        )
    # Put channels at the back
    image = np.moveaxis(image, channels_dimension_index, 3)

    # Get dimension index of second smallest dimension, i.e. time
    second_dimension_index = np.argsort(image.shape)[1]
    # Put time at the front
    image = np.moveaxis(image, second_dimension_index, 0)

    return image


def cell_counter_frame_to_video_frame(
    cell_counter_frame: int, nb_channels=4
) -> int:
    """
    Cell counter index starts at 1, just like Fiji.

    To count frame, it just concatenates all channels.
    For example, with 4 channels, frames 1, 2, 3 and 4 will be frame 1,
    frames 5, 6, 7 and 8 will be frame 2, etc.
    """
    return (cell_counter_frame - 1) // nb_channels
