"""Image tools."""

import operator
from typing import Optional
import numpy as np


def crop_center(img: np.ndarray, bounding: tuple[int]) -> np.ndarray:
    """Crop image to bounding box.

    Parameters
    ----------
    img : np.ndarray
        Image to crop.
    bounding : tuple
        Bounding box to crop to.

    Returns
    -------
    np.ndarray
        Cropped image.
    """
    start = tuple(
        map(lambda a, da: (a - da + 1) // 2, img.shape, bounding)
    )  # 1 is used to match torch method
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]


def get_padding(
    input_shape: tuple[int], output_shape: tuple[int]
) -> tuple[int]:
    """Get padding to apply to input_shape to get output_shape.

    Parameters
    ----------
    input_shape : tuple
        Shape of the input image.
    output_shape : tuple
        Shape of the output image.

    Returns
    -------
    tuple[int]
        Padding to apply to input_shape to get output_shape.
    """
    (_, height, width) = input_shape
    (_, desired_height, desired_width) = output_shape

    # Define margins for both sides
    pad_margin_w_left = abs(desired_width - width) // 2
    pad_margin_w_right = abs(desired_width - width) - pad_margin_w_left

    pad_margin_h_top = abs(desired_height - height) // 2
    pad_margin_h_bottom = abs(desired_height - height) - pad_margin_h_top

    return (
        pad_margin_w_left,
        pad_margin_h_top,
        pad_margin_w_right,
        pad_margin_h_bottom,
    )


def resize_padding(
    image: np.ndarray,
    output_shape: tuple[int],
    mode: str,
    pad_margin_w: Optional[int],
    pad_margin_h: Optional[int],
) -> np.ndarray:
    """Resize image to output_shape with padding.

    Parameters
    ----------
    image : np.ndarray
        Image to resize.
    output_shape : tuple
        Shape of the output image.
    mode : str
        Mode to use for padding. "zero" or "min".
    pad_margin_w : Optional[int]
        Padding margin for width.
    pad_margin_h : Optional[int]
        Padding margin for height.

    Returns
    -------
    np.ndarray
        Resized image with padding.
    """
    channels_to_stack = []
    (depth, height, width) = image.shape
    (_, desired_height, desired_width) = output_shape

    # Crop image if it is too big
    if height > desired_height:
        image = crop_center(image, (depth, desired_height, width))
        height = desired_height
    if width > desired_width:
        image = crop_center(image, (depth, height, desired_width))
        width = desired_width

    raw_padding = get_padding(
        image.shape, output_shape
    )  # left, top, right and bottom

    # Define margins for both sides
    if pad_margin_w is None:
        pad_margin_w_left = raw_padding[0]
        pad_margin_w_right = raw_padding[2]
    else:
        pad_margin_w_left, pad_margin_w_right = pad_margin_w

    if pad_margin_h is None:
        pad_margin_h_top = raw_padding[1]
        pad_margin_h_bottom = raw_padding[3]
    else:
        pad_margin_h_top, pad_margin_h_bottom = pad_margin_h

    # Define padding
    padding = (
        (pad_margin_h_top, pad_margin_h_bottom),
        (pad_margin_w_left, pad_margin_w_right),
    )

    # Pad channels with minimal value of current channel
    for channel_image in range(depth):
        if mode == "min":
            constant_values = np.min(image[channel_image, ...])
        elif mode == "zero":
            constant_values = 0
        else:
            raise ValueError("Unknown mode")
        channel_image = np.pad(
            image[channel_image, ...],
            padding,
            mode="constant",
            constant_values=constant_values,
        )
        channels_to_stack.append(channel_image)

    # Stack padded channels and return
    return np.stack(channels_to_stack)


def resize_image(
    image,
    output_shape: Optional[tuple[int]] = None,
    method="zero",
    pad_margin_w: Optional[int] = None,
    pad_margin_h: Optional[int] = None,
):
    """Resize image to output_shape with padding. Expect image to be a 3D numpy array CYX.
    Output_shape is a tuple CYX.

    Parameters
    ----------
    image : np.ndarray
        Image to resize.
    output_shape : Optional[tuple]
        Shape of the output image.
    method : str
        Method to use for padding. "zero" or "min".
    pad_margin_w : Optional[int]
        Padding margin for width.
    pad_margin_h : Optional[int]
        Padding margin for height.

    Returns
    -------
    np.ndarray
        Resized image with padding.
    """

    if method in ["min", "zero"]:
        # Compute output_shape
        if output_shape is None:
            if pad_margin_h is None or pad_margin_w is None:
                raise ValueError(
                    "If output_shape is None, pad_margin_h and pad_margin_w must be specified"
                )
            output_shape = (
                None,
                sum(pad_margin_h) + image.shape[1],
                sum(pad_margin_w) + image.shape[2],
            )
        # Protect against output_shape and margins inconsistency
        if output_shape is not None and pad_margin_h is not None:
            assert sum(pad_margin_h) == output_shape[1] - image.shape[1]
        if output_shape is not None and pad_margin_w is not None:
            assert sum(pad_margin_w) == output_shape[2] - image.shape[2]
        # Pad to required_dimensions
        return resize_padding(
            image, output_shape, method, pad_margin_w, pad_margin_h
        )

    raise ValueError(f"Unknown resize method: {method}")


def smart_cropping(
    image: np.ndarray,
    margin: int,
    min_x: int,
    min_y: int,
    max_x: Optional[int] = None,
    max_y: Optional[int] = None,
    fade_margin=False,
    pad=False,
) -> np.ndarray:
    """
    Crop microscopy image with smart margin.

    Parameters
    ----------
    image : np.ndarray
        Image to crop. Shape must be (..., H, W).
    margin : int
        Margin to apply.
    min_x : int
        Minimum x coordinate.
    min_y : int
        Minimum y coordinate.
    max_x : Optional[int]
        Maximum x coordinate.
    max_y : Optional[int]
        Maximum y coordinate.
    fade_margin : bool
        If True, fade margin.
    pad : bool
        If True, pad image.

    Returns
    -------
    np.ndarray
        Cropped image.
    """

    # Get image shape
    height, width = image.shape[-2:]

    # Define empty rectangle if single point is given
    if max_x is None:
        max_x = min_x
    if max_y is None:
        max_y = min_y

    # Get smart margin
    clipped_min_y = max(min_y - margin, 0)
    clipped_max_y = min(max_y + margin, height)
    clipped_min_x = max(min_x - margin, 0)
    clipped_max_x = min(max_x + margin, width)

    # Crop image
    clipped_image = np.copy(
        image[..., clipped_min_y:clipped_max_y, clipped_min_x:clipped_max_x]
    )

    # Fade margin
    if fade_margin:
        margin_positions = (
            [
                [y, x]
                for x in range(min_x - clipped_min_x)
                for y in range(clipped_max_y - clipped_min_y)
            ]  # left margin
            + [
                [y, x + max_x - clipped_min_x]
                for x in range(clipped_max_x - max_x)
                for y in range(clipped_max_y - clipped_min_y)
            ]  # right margin
            + [
                [y, x]
                for y in range(min_y - clipped_min_y)
                for x in range(min_x - clipped_min_x, max_x - clipped_min_x)
            ]  # top margin
            + [
                [y + max_y - clipped_min_y, x]
                for y in range(clipped_max_y - max_y)
                for x in range(min_x - clipped_min_x, max_x - clipped_min_x)
            ]  # bottom margin
        )
        # Fade margin
        for margin_position in margin_positions:
            clipped_image[..., margin_position[0], margin_position[1]] = (
                clipped_image[..., margin_position[0], margin_position[1]] / 2
            )

    # Pad if necessary
    if pad and clipped_image.shape[-1] * clipped_image.shape[-2] != (
        max_x - min_x + 2 * margin
    ) * (max_y - min_y + 2 * margin):
        # Compute padding
        pad_margin_h = (
            clipped_min_y - (min_y - margin),
            max_y + margin - clipped_max_y,
        )
        pad_margin_w = (
            clipped_min_x - (min_x - margin),
            max_x + margin - clipped_max_x,
        )
        # Pad
        clipped_image = resize_image(
            clipped_image,
            pad_margin_h=pad_margin_h,
            pad_margin_w=pad_margin_w,
            method="zero",
        )

    return clipped_image


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
