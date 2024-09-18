import concurrent.futures
import time
import numpy as np
import torch
from cellpose import models
from tqdm import tqdm

from ..utils.segmentation_tracking.mask_utils import (
    get_spots_from_frame,
)
from ..utils.cell_spot import CellSpot
from ..utils.cell_track import CellTrack
from ..utils.mid_body_detection.spatial_laptrack import (
    SpatialLapTrack,
)
from ..utils.track_generation import generate_tracks_from_spots


class SegmentationTrackingFactory:
    """Class to perform cell segmentation and tracking.

    Parameters
    ----------
    model_path : str
        Path to the cellpose model
    augment : bool
        cf cellpose documentation
    cellprob_threshold : float
        cf cellpose documentation
    flow_threshold : float
        cf cellpose documentation
    gap_closing_max_distance_ratio : float
        Ratio of average spot size to use for gap closing
    linking_max_distance_ratio : float
        Ratio of average spot size
    max_frame_gap : int
        Maximum number of frames to consider for gap closing
    """

    def __init__(
        self,
        model_path: str,
        augment=True,
        cellprob_threshold=0.0,
        flow_threshold=0.0,
        gap_closing_max_distance_ratio=0.5,
        linking_max_distance_ratio=1,
        max_frame_gap=CellTrack.max_frame_gap,
        minimum_cell_track_length=10,
    ) -> None:
        self.model_path = model_path
        self.augment = augment
        self.cellprob_threshold = cellprob_threshold
        self.flow_threshold = flow_threshold
        self.gap_closing_max_distance_ratio = gap_closing_max_distance_ratio
        self.linking_max_distance_ratio = linking_max_distance_ratio
        self.max_frame_gap = max_frame_gap
        self.minimum_cell_track_length = minimum_cell_track_length

    @staticmethod
    def get_spots_from_cellpose(
        cellpose_results: np.ndarray,
        parallel: bool = False,
    ) -> dict[int, list[CellSpot]]:
        """Extract spots from cellpose results.

        Parameters
        ----------
        cellpose_results : np.ndarray
            TYX
        parallel : bool
            Whether to use parallel processing.

        Returns
        -------
        dict[int, list[CellSpot]]
            Dictionary with frame number as key and list of cell spots as value.
        """
        print("Extracting spots from segmentation results.")
        if parallel:
            future_list = []
            with concurrent.futures.ThreadPoolExecutor() as e:
                for frame, cellpose_result in enumerate(cellpose_results):
                    future_list.append(
                        e.submit(get_spots_from_frame, frame, cellpose_result)
                    )

            cell_dictionary = {
                res.result()[0]: res.result()[1]
                for res in concurrent.futures.as_completed(future_list)
            }
        else:
            cell_dictionary = {}
            for frame, cellpose_result in enumerate(tqdm(cellpose_results)):
                _, spots = get_spots_from_frame(frame, cellpose_result)
                cell_dictionary[frame] = spots

        # Give id number to cell spots
        id_number = 0
        for frame in range(len(cellpose_results)):
            if frame not in cell_dictionary:
                continue
            for cell in cell_dictionary[frame]:
                cell.id = id_number
                id_number += 1

        # Return a sorted dictionary to ensure tracking consistency
        ordered_cell_dictionary = dict(sorted(cell_dictionary.items()))

        return ordered_cell_dictionary

    def perform_segmentation(
        self,
        video: np.ndarray,
    ) -> tuple[np.ndarray, list[np.ndarray], float]:
        """Perform cell segmentation using cellpose.

        Parameters
        ----------
        video : np.ndarray
            TCYX

        Returns
        -------
        np.ndarray
            Cellpose results. TYX.
        list[np.ndarray]
            Cellpose flows.
        float
            Expected diameter of the cells.
        """

        # Cellpose segmentation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.CellposeModel(
            pretrained_model=self.model_path, device=device
        )

        print("Running Cellpose.")
        start = time.time()
        cellpose_results, flows, _ = model.eval(  # TYX
            video,
            channels=[3, 0],
            diameter=0,
            flow_threshold=self.flow_threshold,
            cellprob_threshold=self.cellprob_threshold,
            augment=self.augment,
            resample=False,
        )
        time_second = int(time.time() - start)
        print(f"Done in {time_second} seconds.")

        return cellpose_results, flows, model.diam_labels

    def perform_tracking(
        self, cellpose_results: np.ndarray, diam_labels: float
    ) -> tuple[list[CellSpot], list[CellTrack]]:
        """Perform tracking using laptrack.

        Parameters
        ----------
        cellpose_results : np.ndarray
            TYX
        diam_labels : float
            Expected diameter of the cells.

        Returns
        -------
        list[CellSpot]
            List of cell spots.
        list[CellTrack]
            List of cell tracks.
        """
        cell_spots_dictionary = self.get_spots_from_cellpose(cellpose_results)

        tracking_method = SpatialLapTrack(
            spatial_coord_slice=slice(0, 2),
            spatial_metric="euclidean",
            track_dist_metric="euclidean",
            track_cost_cutoff=diam_labels * self.linking_max_distance_ratio,
            gap_closing_dist_metric="euclidean",
            gap_closing_cost_cutoff=diam_labels
            * self.gap_closing_max_distance_ratio,
            gap_closing_max_frame_count=self.max_frame_gap,
            splitting_cost_cutoff=False,
            merging_cost_cutoff=False,
            alternative_cost_percentile=100,
        )
        cell_tracks = generate_tracks_from_spots(
            cell_spots_dictionary, tracking_method
        )

        # Keep only tracks with a minimum length
        cell_tracks = [
            track
            for track in cell_tracks
            if len(track.spots) >= self.minimum_cell_track_length
        ]

        cell_spots = []
        for frame_spots in cell_spots_dictionary.values():
            cell_spots.extend(frame_spots)

        return cell_spots, cell_tracks

    def perform_segmentation_tracking(
        self,
        video: np.ndarray,
    ) -> tuple[list[CellSpot], list[CellTrack], np.ndarray]:
        """Perform cell segmentation and tracking.

        Parameters
        ----------
        video : np.ndarray
            TCYX

        Returns
        -------
        list[CellSpot]
            List of cell spots.
        list[CellTrack]
            List of cell tracks.
        np.ndarray
            Segmentation results. TYX.
        """

        segmentation_results, _, diam_labels = self.perform_segmentation(video)
        cell_spots, cell_tracks = self.perform_tracking(
            segmentation_results, diam_labels
        )

        return cell_spots, cell_tracks, segmentation_results
