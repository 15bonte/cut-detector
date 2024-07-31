import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from PIL import Image, ImageDraw

from .box_dimensions import BoxDimensions


class BoxDimensionsDln(BoxDimensions):
    """A class to store the dimensions of a box and the delaunay triangulation of the box."""

    def __init__(self):
        super().__init__()
        self.dln = None
        self.list_dln = []
        self.list_points = []  # (x, y)

    def get_list_dln(self):
        """Protection against BoxDimensionsDln old version."""
        if hasattr(self, "list_dln"):
            return self.list_dln
        return [self.dln]

    def get_mask(self, local_shape: tuple[int]) -> np.ndarray:
        """Generate spot mask.

        Parameters
        ----------
        local_shape : tuple
            Shape of the mask.

        Returns
        -------
        np.ndarray
            Mask.
        """
        if hasattr(self, "list_points"):
            mask = np.zeros(local_shape)
            for points in self.list_points:
                binary_image = Image.new(
                    "1", (local_shape[1], local_shape[0]), 0
                )
                draw = ImageDraw.Draw(binary_image)
                draw.polygon(
                    [(point[1], point[0]) for point in points],
                    outline=1,
                    fill=1,
                )
                mask = np.logical_or(mask, np.array(binary_image))
            mask = mask.astype(np.uint8)
            return mask

        indices = np.stack(np.indices(local_shape), axis=-1)
        mask = np.zeros(local_shape)
        for dln in self.get_list_dln():
            out_idx = np.nonzero(dln.find_simplex(indices) + 1)
            mask[out_idx] = 1

        return mask

    def update_attributes(self, relative: bool) -> None:
        """Update attributes

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

        # Compute list of hulls
        for track_frame_points in self.list_points:
            hull = ConvexHull(points=track_frame_points)
            track_frame_points = np.array(track_frame_points)
            self.list_dln.append(Delaunay(track_frame_points[hull.vertices]))

        # Compute hull
        track_frame_points = [
            point
            for track_frame_points in self.list_points
            for point in track_frame_points
        ]
        hull = ConvexHull(points=track_frame_points)
        track_frame_points = np.array(track_frame_points)
        self.dln = Delaunay(track_frame_points[hull.vertices])
