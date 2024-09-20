import numpy as np
from PIL import Image, ImageDraw

from .box_dimensions import BoxDimensions


class BoxDimensionsContour(BoxDimensions):
    """A class to store the dimensions of a box and the precise contour."""

    def __init__(self):
        super().__init__()
        self.list_points = []  # (x, y)

    def get_mask(self, local_shape: tuple[int]) -> np.ndarray:
        """Generate spot mask.

        Parameters
        ----------
        local_shape : tuple
            Shape of the mask.

        Returns
        -------
        np.ndarray
            Mask 2D.
        """
        if not hasattr(self, "list_points"):  # old versions
            assert hasattr(self, "dln")
            list_points = [self.dln.points]
        else:
            list_points = self.list_points

        mask = np.zeros(local_shape)
        for points in list_points:
            binary_image = Image.new("1", (local_shape[1], local_shape[0]), 0)
            draw = ImageDraw.Draw(binary_image)
            draw.polygon(
                [(point[1], point[0]) for point in points],
                outline=1,
                fill=1,
            )
            mask = np.logical_or(mask, np.array(binary_image))
        mask = mask.astype(np.uint8)
        return mask

    def update_list_points(self, relative: bool) -> None:
        """Update list of contour points.

        Parameters
        ----------
        relative : bool
            Relative or absolute coordinates.
        """
        # Switch dimensions
        if relative:
            self.list_points = [
                [
                    [
                        y - self.min_y,
                        x - self.min_x,
                    ]
                    for x, y in track_frame_points
                ]
                for track_frame_points in self.list_points
            ]
        else:
            self.list_points = [
                [[y, x] for x, y in track_frame_points]
                for track_frame_points in self.list_points
            ]
