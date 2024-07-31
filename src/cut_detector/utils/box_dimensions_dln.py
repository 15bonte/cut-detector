import numpy as np
from scipy.spatial import ConvexHull, Delaunay

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

    def get_mask(self, indices, local_shape):
        mask = np.zeros(local_shape)

        for dln in self.get_list_dln():
            out_idx = np.nonzero(dln.find_simplex(indices) + 1)
            mask[out_idx] = 1
        return mask

    def update_attributes(self, relative):
        all_track_frame_points = self.list_points

        # Else, compute convex hull and Delaunay triangulation
        # Switch dimensions
        if relative:
            all_track_frame_points = [
                [
                    [
                        y - self.min_y,
                        x - self.min_x,
                    ]
                    for x, y in track_frame_points
                ]
                for track_frame_points in all_track_frame_points
            ]
        else:
            all_track_frame_points = [
                [[y, x] for x, y in track_frame_points]
                for track_frame_points in all_track_frame_points
            ]

        # Compute list of hulls
        for track_frame_points in all_track_frame_points:
            hull = ConvexHull(points=track_frame_points)
            track_frame_points = np.array(track_frame_points)
            self.list_dln.append(Delaunay(track_frame_points[hull.vertices]))

        # Compute hull
        track_frame_points = [
            point
            for track_frame_points in all_track_frame_points
            for point in track_frame_points
        ]
        hull = ConvexHull(points=track_frame_points)
        track_frame_points = np.array(track_frame_points)
        self.dln = Delaunay(track_frame_points[hull.vertices])
