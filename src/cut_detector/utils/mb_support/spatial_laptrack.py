""" A modified version of LapTrack.
- Maximum distance is only applied to the euclidian distance
"""

import numpy as np
from functools import partial
from typing import Callable, List, cast, Union
from pydantic import Field
from scipy.spatial.distance import cdist
from laptrack import LapTrack, ParallelBackend
import networkx as nx

from laptrack._typing_utils import EdgeType
from laptrack._coo_matrix_builder import coo_matrix_builder
from laptrack._optimization import lap_optimization
from laptrack._cost_matrix import build_frame_cost_matrix


class SpatialLapTrack(LapTrack):
    """
    SpatialLapTrack is a modification of LapTrack where
    track_cost_cutoff and gap_closing_cost_cutoff are compared
    against euclidian spatial distance only.

    (splitting and merging cost cutoff works as usual because
    they are not used in the project for now)

    This is only for filtering out values:
    Optimisation is still run against the chosen metric.
    """

    spatial_coord_slice: slice = Field(
        ..., description="A slice that is used to subset coords"
    )

    spatial_metric: Union[str, Callable] = Field(
        "euclidean",
        description="The metric to use to compute spatial distances",
    )

    class Config:
        arbitrary_types_allowed = True

    def _predict_links(
        self, coords, segment_connected_edges, split_merge_edges
    ) -> nx.Graph:
        """
        Link particles between frames according to the cost function.

        Parameters
        ----------
            coords : List[np.ndarray]
                the input coordinates
            segment_connected_edges : EdgesType
                the connected edges list that will be connected in this step
            split_merge_edges : EdgesType
                the connected edges list that will be connected in split and merge step

        Returns
        -------
            nx.Graph: The resulted tree.

        """
        # initialize tree
        track_tree = nx.Graph()
        for frame, coord in enumerate(coords):
            for j in range(coord.shape[0]):
                track_tree.add_node((frame, j))

        # linking between frames
        edges_list = list(segment_connected_edges) + list(split_merge_edges)

        def _predict_link_single_frame(
            frame: int,
            coord1: np.ndarray,
            coord2: np.ndarray,
        ) -> List[EdgeType]:
            force_end_indices = [
                e[0][1] for e in edges_list if e[0][0] == frame
            ]
            force_start_indices = [
                e[1][1] for e in edges_list if e[1][0] == frame + 1
            ]

            dist_matrix = cdist(coord1, coord2, metric=self.track_dist_metric)
            dist_matrix[force_end_indices, :] = np.inf
            dist_matrix[:, force_start_indices] = np.inf

            # While this has not been checked, it is highly likely
            # that coord are ordered the way specified in predict_dataframe
            spatial_coord1 = coord1[:, self.spatial_coord_slice]
            spatial_coord2 = coord2[:, self.spatial_coord_slice]

            spatial_dist_matrix = cdist(
                spatial_coord1, spatial_coord2, metric=self.spatial_metric
            )
            spatial_dist_matrix[force_end_indices, :] = np.inf
            spatial_dist_matrix[:, force_start_indices] = np.inf

            # ind = np.where(dist_matrix < self.track_cost_cutoff)
            ind = np.where(spatial_dist_matrix < self.track_cost_cutoff)
            dist_matrix = coo_matrix_builder(  # keeping the true score here
                dist_matrix.shape,
                row=ind[0],
                col=ind[1],
                data=dist_matrix[(*ind,)],
                dtype=dist_matrix.dtype,
            )

            cost_matrix = build_frame_cost_matrix(
                dist_matrix,
                track_start_cost=self.track_start_cost,
                track_end_cost=self.track_end_cost,
            )
            xs, _ = lap_optimization(cost_matrix)

            count1 = dist_matrix.shape[0]
            count2 = dist_matrix.shape[1]
            connections = [(i, xs[i]) for i in range(count1) if xs[i] < count2]
            edges: List[EdgeType] = [
                ((frame, connection[0]), (frame + 1, connection[1]))
                for connection in connections
            ]
            return edges

        if self.parallel_backend == ParallelBackend.serial:
            all_edges = []
            for frame, (coord1, coord2) in enumerate(
                zip(coords[:-1], coords[1:])
            ):
                edges = _predict_link_single_frame(frame, coord1, coord2)
                all_edges.extend(edges)
        elif self.parallel_backend == ParallelBackend.ray:
            try:
                import ray  # type: ignore (removes the import warning)
            except ImportError:
                raise ImportError(
                    "Please install `ray` to use `ParallelBackend.ray`."
                )
            remote_func = ray.remote(_predict_link_single_frame)
            res = [
                remote_func.remote(frame, coord1, coord2)
                for frame, (coord1, coord2) in enumerate(
                    zip(coords[:-1], coords[1:])
                )
            ]
            all_edges = sum(ray.get(res), [])
        else:
            raise ValueError(
                f"Unknown parallel backend {self.parallel_backend}. "
                + f"Must be one of {', '.join([ps.name for ps in ParallelBackend])}."
            )

        track_tree.add_edges_from(all_edges)
        track_tree.add_edges_from(segment_connected_edges)
        return track_tree

    def _get_gap_closing_matrix(
        self, segments_df, *, force_end_nodes=[], force_start_nodes=[]
    ):
        """
        Generate the cost matrix for connecting segment ends.

        Parameters
        ----------
        segments_df : pd.DataFrame
            must have the columns "first_frame", "first_index", "first_crame_coords", "last_frame", "last_index", "last_frame_coords"
        force_end_nodes : list of int
            the indices of the segments_df that is forced to be end for future connection
        force_start_nodes : list of int
            the indices of the segments_df that is forced to be start for future connection

        Returns
        -------
        segments_df: pd.DataFrame
            the segments dataframe with additional column "gap_closing_candidates"
            (index of the candidate row of segments_df, the associated costs)
         gap_closing_dist_matrix: coo_matrix_builder
            the cost matrix for gap closing candidates

        """
        if self.gap_closing_cost_cutoff:

            def to_gap_closing_candidates(row, segments_df):
                # if the index is in force_end_indices, do not add to gap closing candidates
                if (row["last_frame"], row["last_index"]) in force_end_nodes:
                    return [], []

                target_coord = row["last_frame_coords"]
                target_spatial_coord = target_coord[self.spatial_coord_slice]
                frame_diff = segments_df["first_frame"] - row["last_frame"]

                # only take the elements that are within the frame difference range.
                # segments in df is later than the candidate segment (row)
                indices = (1 <= frame_diff) & (
                    frame_diff <= self.gap_closing_max_frame_count
                )
                df = segments_df[indices]
                force_start = df.apply(
                    lambda row: (row["first_frame"], row["first_index"])
                    in force_start_nodes,
                    axis=1,
                )
                df = df[~force_start]
                # do not connect to the segments that is forced to be start
                # note: can use KDTree if metric is distance,
                # but might not be appropriate for general metrics
                # https://stackoverflow.com/questions/35459306/find-points-within-cutoff-distance-of-other-points-with-scipy # noqa
                # TrackMate also uses this (trivial) implementation.
                if len(df) > 0:
                    target_dist_matrix = cdist(
                        [target_coord],
                        np.stack(df["first_frame_coords"].values),
                        metric=self.gap_closing_dist_metric,
                    )
                    assert target_dist_matrix.shape[0] == 1

                    # spatial distance
                    spatial_target_dist_matrix = cdist(
                        [target_spatial_coord],
                        # np.stack([df["first_frame_coords"].values[0][self.spatial_coord_slice]]),
                        np.stack(df["first_frame_coords"].values)[
                            :, self.spatial_coord_slice
                        ],
                        metric=self.spatial_metric,
                    )
                    assert spatial_target_dist_matrix.shape[0] == 1

                    indices2 = np.where(
                        # target_dist_matrix[0] < self.gap_closing_cost_cutoff
                        spatial_target_dist_matrix[0]
                        < self.gap_closing_cost_cutoff
                    )[0]
                    return (
                        df.index[indices2].values,
                        target_dist_matrix[0][indices2],
                    )
                else:
                    return [], []

            if self.parallel_backend == ParallelBackend.serial:
                segments_df["gap_closing_candidates"] = segments_df.apply(
                    partial(
                        to_gap_closing_candidates, segments_df=segments_df
                    ),
                    axis=1,
                )
            elif self.parallel_backend == ParallelBackend.ray:
                try:
                    import ray  # type: ignore
                except ImportError:
                    raise ImportError(
                        "Please install `ray` to use `ParallelBackend.ray`."
                    )
                remote_func = ray.remote(to_gap_closing_candidates)
                segments_df_id = ray.put(segments_df)
                res = [
                    remote_func.remote(row, segments_df_id)
                    for _, row in segments_df.iterrows()
                ]
                segments_df["gap_closing_candidates"] = ray.get(res)
            else:
                raise ValueError(
                    f"Unknown parallel_backend {self.parallel_backend}. "
                )
        else:
            segments_df["gap_closing_candidates"] = [([], [])] * len(
                segments_df
            )

        N_segments = len(segments_df)
        gap_closing_dist_matrix = coo_matrix_builder(
            (N_segments, N_segments), dtype=np.float32
        )
        for ind, row in segments_df.iterrows():
            candidate_inds = row["gap_closing_candidates"][0]
            candidate_costs = row["gap_closing_candidates"][1]
            # row ... track end, col ... track start
            gap_closing_dist_matrix[(int(cast(int, ind)), candidate_inds)] = (
                candidate_costs
            )

        return segments_df, gap_closing_dist_matrix
