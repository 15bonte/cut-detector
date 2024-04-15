""" Modified version of LapTrack, with spatial parameters
and additional debug options
"""

import networkx as nx
import numpy as np

from typing import Callable, List, cast, Any, Dict, Union
from numpy.typing import NDArray
from pydantic import Field
from scipy.spatial.distance import cdist
from laptrack import LapTrack, ParallelBackend
from functools import partial

from laptrack._typing_utils import NumArray, Int, EdgeType
from laptrack._coo_matrix_builder import coo_matrix_builder
from laptrack._optimization import lap_optimization
from laptrack._cost_matrix import build_frame_cost_matrix, build_segment_cost_matrix

from cut_detector.factories.mb_support.tracking import SpatialLapTrack

class SpatialLaptrackDebug(LapTrack):
    """ SpatialLapTrack is a modification of LapTrack where
    track_cost_cutoff and gap_closing_cost_cutoff are compared
    against euclidian spatial distance only.

    (splitting and merging cost cutoff works as usual because
    they are not used in the project for now)

    This is only for filtering out values: 
    Optimisation is still run against the chosen metric.

    Additionnaly this version provides debug/visualization
    parameters.
    """

    spatial_coord_slice: slice = Field(
        slice(0,2),
        description="A slice that is used to subset coords"
    )

    spatial_metric: Union[str, Callable]  = Field(
        "euclidean",
        description="The metric to use to compute spatial distances"
    )

    show_predict_link_debug: bool = Field(
        False,
        description="Enables the debugging on the 1st part of the tracking"
    )

    show_gap_closing_debug: bool = Field(
        False,
        description="Enables the debugging on the 2nd part of the tracking"
    )

    class Config:
        arbitrary_types_allowed = True

    def import_settings_from_lt(self, lt: LapTrack):
        """ Import settings from a LapTrack instance
        """
        self.track_dist_metric           = lt.track_dist_metric
        self.track_cost_cutoff           = lt.track_cost_cutoff
        self.gap_closing_dist_cutoff     = lt.gap_closing_dist_cutoff
        self.gap_closing_cost_cutoff     = lt.gap_closing_cost_cutoff
        self.gap_closing_max_frame_count = lt.gap_closing_max_frame_count
        self.splitting_dist_metric       = lt.splitting_dist_metric
        self.splitting_cost_cutoff       = lt.splitting_cost_cutoff
        self.merging_dist_metric         = lt.merging_dist_metric
        self.merging_cost_cutoff         = lt.merging_cost_cutoff
        self.track_start_cost            = lt.track_start_cost
        self.track_end_cost              = lt.track_end_cost
        self.segment_start_cost          = lt.segment_start_cost
        self.segment_end_cost            = lt.segment_end_cost
        self.no_splitting_cost           = lt.no_splitting_cost
        self.no_merging_cost             = lt.no_merging_cost
        self.alternative_cost_factor     = lt.alternative_cost_factor
        self.alternative_cost_percentile = lt.alternative_cost_percentile
        self.alternative_cost_percentile_interpolation = lt.alternative_cost_percentile_interpolation
        self.parallel_backend            = lt.parallel_backend


    def import_settings_from_slt(self, slt: SpatialLapTrack):
        """ Import settings from a SpatialLapTrack instance
        """
        # Importing base settings first,
        # then settings from Spatial LapTrack
        self.import_settings_from_lt(slt)
        self.spatial_coord_slice = slt.spatial_coord_slice
        self.spatial_metric = slt.spatial_metric


    def _predict_links(
        self, coords: List[NDArray], segment_connected_edges, split_merge_edges
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
            if self.show_predict_link_debug:
                print(f"f{frame}: {coord}")
            for j in range(coord.shape[0]):
                track_tree.add_node((frame, j))

        # linking between frames
        edges_list = list(segment_connected_edges) + list(split_merge_edges)
        if self.show_predict_link_debug:
            print("edges_list:", edges_list)

        if self.show_predict_link_debug:
            print("@KEY_FRAME_LINK@")


        def _predict_link_single_frame(
            frame: int,
            coord1: np.ndarray,
            coord2: np.ndarray,
        ) -> List[EdgeType]:
            
            print(f"\n\n--Predicting Link Frame {frame}--")

            if self.show_predict_link_debug:
                print(f"\nCoordinates:\nc1 {coord1}\nc2 {coord2}")

            force_end_indices = [e[0][1] for e in edges_list if e[0][0] == frame]
            force_start_indices = [e[1][1] for e in edges_list if e[1][0] == frame + 1]
            

            dist_matrix = cdist(coord1, coord2, metric=self.track_dist_metric)
            dist_matrix[force_end_indices, :] = np.inf
            dist_matrix[:, force_start_indices] = np.inf

            if self.show_predict_link_debug:
                print(f"\nDist Matrix\n{dist_matrix}")

            # While this has not been checked, it is highly likely
            # that coord are ordered the way specified in predict_dataframe
            spatial_coord1 = coord1[:,self.spatial_coord_slice]
            spatial_coord2 = coord2[:,self.spatial_coord_slice]

            spatial_dist_matrix = cdist(
                spatial_coord1, 
                spatial_coord2, 
            metric=self.spatial_metric)
            
            # Convert NaN to track_cost_cutoff*2 (to be consisten t with dist_matrix)
            spatial_dist_matrix = np.nan_to_num(spatial_dist_matrix, nan=self.track_cost_cutoff*2)

            spatial_dist_matrix[force_end_indices, :] = np.inf
            spatial_dist_matrix[:, force_start_indices] = np.inf

            if self.show_predict_link_debug:
                print(f"\nSpatial Dist Matrix:\n{spatial_dist_matrix}")

            # returns a tuple(row, col)
            # where row and col are np arrays
            # with the indices of the element satisfying the condition
            # ([r1, r2, rn...], [c1, c2, cn])
            ind = np.where(spatial_dist_matrix < self.track_cost_cutoff)
            print(f"\nIndicies below spatial cutoff:\n", ind)
            print(f"\nCOO mat data:\n{dist_matrix[(*ind,)]}")
            dist_matrix = coo_matrix_builder( # keeping the true score here
                dist_matrix.shape,
                row=ind[0],
                col=ind[1],
                data=dist_matrix[(*ind,)],
                dtype=dist_matrix.dtype,
            )

            # To better understand the way this matrix is represented, see:
            # https://ransakaravihara.medium.com/sparse-matrices-what-when-and-why-b2271af1fd68
            # basically the matrix may have a lot of 0s
            # Hence the name "sparse" matrix.
            # There are several way to represent it
            # One of them is COO = COOrdinate list
            # another is CSR = Compressed Sparse Row
            # lap_optimization uses LIL = LIst of Lists

            cost_matrix = build_frame_cost_matrix(
                dist_matrix,
                track_start_cost=self.track_start_cost,
                track_end_cost=self.track_end_cost,
            )
            if self.show_predict_link_debug:
                print(f"\nDist metric cost matrix:", cost_matrix, sep="\n")
                # densifying the sparse matrix to print a real matrix (with 0s)
                print(f"\nDist Metric Cost Matrix (dense mode):\n{cost_matrix.todense()}")

            xs, ys = lap_optimization(cost_matrix)
            if self.show_predict_link_debug:
                print(f"\nLap xs:{xs}\nys:{ys}\n")

            count1 = dist_matrix.shape[0]
            count2 = dist_matrix.shape[1]
            if self.show_predict_link_debug:
                print("count1 (dist_matrix.shape 0):", count1)
                print("count2 (dist_matrix.shape 1):", count2)
                print("new dist_matrix:\n", dist_matrix.data)
                for i in range(count1):
                    added_conn = "/"
                    if xs[i] < count2:
                        added_conn = (i, xs[i])
                    print(f"i:{i} | 'xs[i]:{xs[i]} < count2:{count2}' -> {added_conn}")
            connections = [(i, xs[i]) for i in range(count1) if xs[i] < count2]
            edges: List[EdgeType] = [
                ((frame, connection[0]), (frame + 1, connection[1]))
                for connection in connections
            ]
            if self.show_predict_link_debug:
                print(f"frame {frame}: edges:\n{edges}")
            return edges

        if self.parallel_backend == ParallelBackend.serial:
            all_edges = []
            for frame, (coord1, coord2) in enumerate(zip(coords[:-1], coords[1:])):
                # coords[:-1] = all but last one
                # coords[1:] = all but first one
                # zip(...): pair of current/next frame points (because coords are several points)
                edges = _predict_link_single_frame(frame, coord1, coord2)
                all_edges.extend(edges)
        elif self.parallel_backend == ParallelBackend.ray:
            try:
                import ray # type: ignore (removes the import warning)
            except ImportError:
                raise ImportError("Please install `ray` to use `ParallelBackend.ray`.")
            remote_func = ray.remote(_predict_link_single_frame)
            res = [
                remote_func.remote(frame, coord1, coord2)
                for frame, (coord1, coord2) in enumerate(zip(coords[:-1], coords[1:]))
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

            if self.show_gap_closing_debug:
                print("@KEY_GAP_CLOSING@")

                print("\n\n--- Starting Gap Closing ---")
                print("\nsegments_df:\n", segments_df)

            def to_gap_closing_candidates(row, segments_df):

                if self.show_gap_closing_debug:
                    print("\nGap closing row:\n", row)

                # if the index is in force_end_indices, do not add to gap closing candidates
                if (row["last_frame"], row["last_index"]) in force_end_nodes:
                    return [], []

                target_coord = row["last_frame_coords"]
                target_spatial_coord = target_coord[self.spatial_coord_slice]
                frame_diff = segments_df["first_frame"] - row["last_frame"]

                if self.show_gap_closing_debug:
                    print(
                        "\nTarget coord:", target_coord, 
                        "Target spatial coord:", target_spatial_coord,
                        f"frame_diff (max:{self.gap_closing_max_frame_count}):", frame_diff,
                        sep="\n",
                    )

                # only take the elements that are within the frame difference range.
                # segments in df is later than the candidate segment (row)
                indices = (1 <= frame_diff) & (
                    frame_diff <= self.gap_closing_max_frame_count
                )
                df = segments_df[indices]

                if self.show_gap_closing_debug:
                    print("\nValid segments_df indicies (valid: frame_diff>=1 \
                          and frame_diff <= frame_cutoff):\n{indices}")
                    print("\ndf:\n{df}")
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

                    if self.show_gap_closing_debug:
                        print("\nTarget dist matrix:\n", target_dist_matrix)

                    # spatial distance
                    spatial_target_dist_matrix = cdist(
                        [target_spatial_coord],
                        # np.stack([df["first_frame_coords"].values[0][self.spatial_coord_slice]]),
                        np.stack(df["first_frame_coords"].values)[:,self.spatial_coord_slice],
                        metric=self.spatial_metric,
                    )
                    assert spatial_target_dist_matrix.shape[0] == 1

                    if self.show_gap_closing_debug:
                        print("\nTarget spatial matrix:\n", spatial_target_dist_matrix)
                        print("Gap Closing cuttoff:", self.gap_closing_cost_cutoff)

                    indices2 = np.where(
                        # target_dist_matrix[0] < self.gap_closing_cost_cutoff
                        spatial_target_dist_matrix[0] < self.gap_closing_cost_cutoff
                    )[0]

                    if self.show_gap_closing_debug:
                        print("Indicies2 (spatial_dist) below cutoff:\n", indices2)

                        print(
                            "\nReturning:",
                            "df values at indices2:", df.index[indices2].values,
                            "'target_dist_matrix[0][indices2]:'", target_dist_matrix[0][indices2]
                        )

                    return (
                        df.index[indices2].values,
                        target_dist_matrix[0][indices2],
                    )
                else:
                    return [], []

            if self.parallel_backend == ParallelBackend.serial:
                segments_df["gap_closing_candidates"] = segments_df.apply(
                    partial(to_gap_closing_candidates, segments_df=segments_df), axis=1
                )
                if self.show_gap_closing_debug:
                    print("\nObtained modified segments_df:\n", segments_df)
            elif self.parallel_backend == ParallelBackend.ray:
                try:
                    import ray # type: ignore
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
                raise ValueError(f"Unknown parallel_backend {self.parallel_backend}. ")
        else:
            segments_df["gap_closing_candidates"] = [([], [])] * len(segments_df)

        N_segments = len(segments_df)
        gap_closing_dist_matrix = coo_matrix_builder(
            (N_segments, N_segments), dtype=np.float32
        )
        for ind, row in segments_df.iterrows():
            candidate_inds = row["gap_closing_candidates"][0]
            candidate_costs = row["gap_closing_candidates"][1]
            # row ... track end, col ... track start
            gap_closing_dist_matrix[
                (int(cast(int, ind)), candidate_inds)
            ] = candidate_costs

        if self.show_gap_closing_debug:
            print("@KEY_GAP_CLOSING_DIST_MAT@")
            print("\nGap Closing dist matrix (dense):\n", gap_closing_dist_matrix.to_coo_matrix().todense())

        return segments_df, gap_closing_dist_matrix
    



    def _link_gap_split_merge_from_matrix(
        self,
        segments_df, # for resolution
        track_tree, # inout return
        gap_closing_dist_matrix, # << The one we are interested in
        splitting_dist_matrix, # probably empty
        merging_dist_matrix, # probably empty
        splitting_all_candidates, # probably empty
        merging_all_candidates, # probably empty
    ):
        if self.show_gap_closing_debug:
            print("@KEY_FRAME_GSM_MATRIX@")
            print(
                "\n--- LinkGapSplitMerge other parameters ---"
                "splitting dist matrix:", splitting_dist_matrix.to_coo_matrix().todense(),
                "merging dist matrix", merging_dist_matrix.to_coo_matrix().todense(),
                "splitting_all_candidates", splitting_all_candidates,
                "merging_all_candidates", merging_all_candidates,
                sep="\n"
            )

        cost_matrix = build_segment_cost_matrix(
            gap_closing_dist_matrix,
            splitting_dist_matrix,
            merging_dist_matrix,
            self.segment_start_cost,
            self.segment_end_cost,
            self.no_splitting_cost,
            self.no_merging_cost,
            self.alternative_cost_factor,
            self.alternative_cost_percentile,
            self.alternative_cost_percentile_interpolation,
        )

        if self.show_gap_closing_debug:
            print("\nGapSplitMerge Cost Matrix:", cost_matrix, sep="\n")
            old_options = np.get_printoptions()
            np.set_printoptions(linewidth=np.inf)
            print("\nSame but in dense mode:", cost_matrix.todense(), sep="\n")
            np.set_printoptions(**old_options)

        if not cost_matrix is None:
            # FIXME connected_edges_list

            xs, ys = lap_optimization(cost_matrix)
            if self.show_gap_closing_debug:
                print(f"\nLap Optimized:\nxs: {xs}\nys: {ys}")

            M = gap_closing_dist_matrix.shape[0]
            N1 = splitting_dist_matrix.shape[1]
            N2 = merging_dist_matrix.shape[1]

            if self.show_gap_closing_debug:
                print(f"\nM:{M}\nN1:{N1}\nN2:{N2}")

                print("col_ind Thresholds:")
                print(f"idx < M({M}): Gap Closing")
                print(f"M({M}) <= idx < M+N2({M+N2}): Merging")
                print(f"M({M}) <= idx < M+N1({M+N1}): Splitting")
                print(f"idx >= max(M+N1, M+N2)({max(M+N1, M+N2)}): Dropped")

                print(f"\nReminder: segments_df:\n{segments_df}")

            for ind, row in segments_df.iterrows():
                col_ind = xs[ind]
                if self.show_gap_closing_debug:
                    print(f"\n-- row --\n{row}\n")
                    print("-> col_ind (xs[ind]):", col_ind)
                first_frame_index = (row["first_frame"], row["first_index"])
                last_frame_index = (row["last_frame"], row["last_index"])
                printed = False
                if col_ind < M:
                    target_frame_index = tuple(
                        segments_df.loc[col_ind, ["first_frame", "first_index"]]
                    )
                    track_tree.add_edge(last_frame_index, target_frame_index)
                    print(f"    > Gap Closing: last frame idx:{last_frame_index} target frame idx:{target_frame_index}")
                    printed = True
                elif col_ind < M + N2:
                    print("    > Merging")
                    track_tree.add_edge(
                        last_frame_index,
                        tuple(merging_all_candidates[col_ind - M]),
                    )
                    printed = True

                row_ind = ys[ind]
                if M <= row_ind and row_ind < M + N1:
                    print("    > Splitting")
                    track_tree.add_edge(
                        first_frame_index,
                        tuple(splitting_all_candidates[row_ind - M]),
                    )
                    printed = True

                if not printed:
                    print("    > Dropped")

        if self.show_predict_link_debug or self.show_gap_closing_debug:
            print("\n\n\n----- LapTrack Done -----\n\n\n\n")

        return track_tree

