import numpy as np


def zero_to_bottom_right(image: np.ndarray) -> np.ndarray:
    """Set to zero the bottom right part of the image.

    Parameters
    ----------
    image : np.ndarray
        The image to augment.

    Returns
    -------
    np.ndarray
        The augmented image.
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


def zero_to_top_left(image: np.ndarray) -> np.ndarray:
    """Set to zero the top left part of the image.

    Parameters
    ----------
    image : np.ndarray
        The image to augment.

    Returns
    -------
    np.ndarray
        The augmented image.
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


def zero_to_bottom_left(image: np.ndarray) -> np.ndarray:
    """
    Set to zero the bottom left part of the image.

    Parameters
    ----------
    image : np.ndarray
        The image to augment.

    Returns
    -------
    np.ndarray
        The augmented image.
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


def zero_to_top_right(image: np.ndarray) -> np.ndarray:
    """Set to zero the top right part of the image.

    Parameters
    ----------
    image : np.ndarray
        The image to augment.

    Returns
    -------
    np.ndarray
        The augmented image.
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
    """Manage the augmentation of images for microtubules detection."""

    def __init__(self):
        self.augmentations = self.merge_augmentations()

    @classmethod
    def merge_augmentations(cls) -> dict[str, int]:
        """
        Merge values given to augmentation categories by different peaks.

        Returns
        -------
        dict[str, int]
            The merged augmentations.
        """
        augmentations = {}
        for category in [
            "top",
            "bottom",
            "left",
            "right",
            "top_left",
            "bottom_right",
            "top_right",
            "bottom_left",
        ]:
            value = 0  # by default, no MT is seen
            if value is not None:
                augmentations[category] = value
        return augmentations

    def generate_augmentations(
        self, original_image
    ) -> dict[str, dict[int, np.ndarray]]:
        """
        Generate image augmentations from the given image.

        Parameters
        ----------
        image: np.ndarray
            The image to augment. CYX.

        Returns
        -------
        dict[str, dict[int, np.ndarray]]
            The augmented images and their category.
        """
        image = np.moveaxis(original_image, 0, -1)  # YXC
        assert image.shape[0] == image.shape[1]

        augmentations = {}

        if "top" in self.augmentations:
            transformed_image = np.copy(image)
            top_image = image[: image.shape[0] - image.shape[0] // 2, :]
            reversed_top_image = top_image[::-1, :]
            transformed_image[image.shape[0] // 2 :, :] = reversed_top_image
            augmentations["top"] = {
                "image": transformed_image,
                "category": self.augmentations["top"],
            }

        if "bottom" in self.augmentations:
            transformed_image = np.copy(image)
            bottom_image = image[image.shape[0] // 2 :, :]
            reversed_bottom_image = bottom_image[::-1, :]
            transformed_image[: image.shape[0] - image.shape[0] // 2, :] = (
                reversed_bottom_image
            )
            augmentations["bottom"] = {
                "image": transformed_image,
                "category": self.augmentations["bottom"],
            }

        if "left" in self.augmentations:
            transformed_image = np.copy(image)
            left_image = image[:, : image.shape[1] - image.shape[1] // 2]
            reversed_left_image = left_image[:, ::-1]
            transformed_image[:, image.shape[1] // 2 :] = reversed_left_image
            augmentations["left"] = {
                "image": transformed_image,
                "category": self.augmentations["left"],
            }

        if "right" in self.augmentations:
            transformed_image = np.copy(image)
            right_image = image[:, image.shape[1] // 2 :]
            reversed_right_image = right_image[:, ::-1]
            transformed_image[:, : image.shape[1] - image.shape[1] // 2] = (
                reversed_right_image
            )
            augmentations["right"] = {
                "image": transformed_image,
                "category": self.augmentations["right"],
            }

        if "top_left" in self.augmentations:
            transformed_image = zero_to_bottom_right(image) + np.moveaxis(
                zero_to_bottom_right(image)[::-1, ::-1].transpose(), 0, -1
            )
            augmentations["top_left"] = {
                "image": transformed_image,
                "category": self.augmentations["top_left"],
            }

        if "bottom_right" in self.augmentations:
            transformed_image = zero_to_top_left(image) + np.moveaxis(
                zero_to_top_left(image)[::-1, ::-1].transpose(), 0, -1
            )
            augmentations["bottom_right"] = {
                "image": transformed_image,
                "category": self.augmentations["bottom_right"],
            }

        if "top_right" in self.augmentations:
            transformed_image = zero_to_bottom_left(image) + np.moveaxis(
                zero_to_bottom_left(image).transpose(), 0, -1
            )
            augmentations["top_right"] = {
                "image": transformed_image,
                "category": self.augmentations["top_right"],
            }

        if "bottom_left" in self.augmentations:
            transformed_image = zero_to_top_right(image) + np.moveaxis(
                zero_to_top_right(image).transpose(), 0, -1
            )
            augmentations["bottom_left"] = {
                "image": transformed_image,
                "category": self.augmentations["bottom_left"],
            }

        return augmentations
