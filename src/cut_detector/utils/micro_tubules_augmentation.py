from typing import Optional

import numpy as np

from .peak import Peak


def zero_to_bottom_right(image):
    """
    Set to zero the bottom right part of the image.
    """
    image = np.copy(image)
    square_shape = image.shape[0]
    for i in range(square_shape):
        for j in range(square_shape):
            if i + j + 1 > square_shape:
                image[i, j] = 0
            if i + j + 1 == square_shape:  # diagonal
                image[i, j] = image[i, j] / 2
    return image


def zero_to_top_left(image):
    """
    Set to zero the top left part of the image.
    """
    image = np.copy(image)
    square_shape = image.shape[0]
    for i in range(square_shape):
        for j in range(square_shape):
            if i + j + 1 < square_shape:
                image[i, j] = 0
            if i + j + 1 == square_shape:  # diagonal
                image[i, j] = image[i, j] / 2
    return image


def zero_to_bottom_left(image):
    """
    Set to zero the bottom left part of the image.
    """
    image = np.copy(image)
    square_shape = image.shape[0]
    for i in range(square_shape):
        for j in range(square_shape):
            if i - j > 0:
                image[i, j] = 0
            if i - j == 0:  # diagonal
                image[i, j] = image[i, j] / 2
    return image


def zero_to_top_right(image):
    """
    Set to zero the top right part of the image.
    """
    image = np.copy(image)
    square_shape = image.shape[0]
    for i in range(square_shape):
        for j in range(square_shape):
            if i - j < 0:
                image[i, j] = 0
            if i - j == 0:  # diagonal
                image[i, j] = image[i, j] / 2
    return image


class MicroTubulesAugmentation:
    """
    Manage the augmentation of images for microtubules detection.
    """

    def __init__(self, peaks: Optional[list[Peak]] = None):
        self.augmentations = Peak.enabled_augmentation(peaks)

    def generate_augmentations(self, image) -> dict[str, np.ndarray]:
        """
        Generate image augmentations from the given image.

        Parameters
        ----------
        image: np.ndarray
            The image to augment. YXC

        """
        assert image.shape[0] == image.shape[1]

        augmentations = {}

        if "top" in self.augmentations:
            transformed_image = np.copy(image)
            top_image = image[: image.shape[0] - image.shape[0] // 2, :]
            reversed_top_image = top_image[::-1, :]
            transformed_image[image.shape[0] // 2 :, :] = reversed_top_image
            augmentations["top"] = transformed_image

        if "bottom" in self.augmentations:
            transformed_image = np.copy(image)
            bottom_image = image[image.shape[0] // 2 :, :]
            reversed_bottom_image = bottom_image[::-1, :]
            transformed_image[: image.shape[0] - image.shape[0] // 2, :] = (
                reversed_bottom_image
            )
            augmentations["bottom"] = transformed_image

        if "left" in self.augmentations:
            transformed_image = np.copy(image)
            left_image = image[:, : image.shape[1] - image.shape[1] // 2]
            reversed_left_image = left_image[:, ::-1]
            transformed_image[:, image.shape[1] // 2 :] = reversed_left_image
            augmentations["left"] = transformed_image

        if "right" in self.augmentations:
            transformed_image = np.copy(image)
            right_image = image[:, image.shape[1] // 2 :]
            reversed_right_image = right_image[:, ::-1]
            transformed_image[:, : image.shape[1] - image.shape[1] // 2] = (
                reversed_right_image
            )
            augmentations["right"] = transformed_image

        if "top_left" in self.augmentations:
            transformed_image = zero_to_bottom_right(image) + np.moveaxis(
                zero_to_bottom_right(image)[::-1, ::-1].transpose(), 0, -1
            )
            augmentations["top_left"] = transformed_image

        if "bottom_right" in self.augmentations:
            transformed_image = zero_to_top_left(image) + np.moveaxis(
                zero_to_top_left(image)[::-1, ::-1].transpose(), 0, -1
            )
            augmentations["bottom_right"] = transformed_image

        if "top_right" in self.augmentations:
            transformed_image = zero_to_bottom_left(image) + np.moveaxis(
                zero_to_bottom_left(image).transpose(), 0, -1
            )
            augmentations["top_right"] = transformed_image

        if "bottom_left" in self.augmentations:
            transformed_image = zero_to_top_right(image) + np.moveaxis(
                zero_to_top_right(image).transpose(), 0, -1
            )
            augmentations["bottom_left"] = transformed_image

        return augmentations
