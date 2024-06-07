import os
import sys
import pickle
import numpy as np
import torch
import imagej
import scyjava as sj
from cellpose import models
from ..utils.cell_spot import CellSpot
from ..utils.cell_track import CellTrack
from scipy.spatial import ConvexHull

from ..utils.mb_support.tracking.spatial_laptrack import (
    SpatialLapTrack,
)
from ..utils.trackmate_track import TrackMateTrack
from ..utils.trackmate_spot import TrackMateSpot
from ..utils.gen_track import generate_tracks_from_spots


# Finding barycenters of each cell
def barycenter(cellpose_results, frame):
    max = np.max(cellpose_results[frame])
    mean_x = []
    mean_y = []
    cells_per_frame = []
    for i in range(1, max + 1):
        cell_indices = np.where(cellpose_results[frame] == i)
        if len(cell_indices[0]) == 0 or len(cell_indices[1]) == 0:
            break
        Sx = np.sum(cell_indices[1])
        Sy = np.sum(cell_indices[0])
        mean_x.append(Sx / len(cell_indices[1]))
        mean_y.append(Sy / len(cell_indices[0]))
        cells_per_frame.append(cell_indices)
    return (mean_x, mean_y, cells_per_frame)


def load_tracks_and_spots(
    trackmate_tracks_path: str, spots_path: str
) -> tuple[list[TrackMateTrack], list[TrackMateSpot]]:
    """
    Load saved spots and tracks generated from Trackmate xml file.
    """
    trackmate_tracks: list[TrackMateTrack] = []
    for track_file in os.listdir(trackmate_tracks_path):
        with open(os.path.join(trackmate_tracks_path, track_file), "rb") as f:
            trackmate_track: TrackMateTrack = pickle.load(f)
            trackmate_track.adapt_deprecated_attributes()
            trackmate_tracks.append(trackmate_track)

    spots: list[TrackMateSpot] = []
    for spot_file in os.listdir(spots_path):
        with open(os.path.join(spots_path, spot_file), "rb") as f:
            spots.append(pickle.load(f))

    return trackmate_tracks, spots


class SegmentationTrackingFactory:
    """
    Class to perform cell segmentation and tracking.

    Args:
        model_path (str): path to the cellpose model
        augment (bool)
        cellprob_threshold (float)
        flow_threshold (float)
        gap_closing_max_distance_ratio (float): ratio of average spot size
        linking_max_distance_ratio  (float): ratio of average spot size
        max_frame_gap (int)
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

    def perform_trackmate_tracking(
        self,
        video_path: str,
        fiji_path: str,
        save_folder: str,
        fast_mode: bool,
    ):
        """
        Use Trackmate to run both segmentation and tracking.
        """
        ij_instance = imagej.init(fiji_path, mode="interactive")

        bf_plugin = sj.jimport("loci.plugins.BF")
        importer_options = sj.jimport("loci.plugins.in.ImporterOptions")

        # Import TrackMate via scyjava
        track_mate_model = sj.jimport(
            "fiji.plugin.trackmate.Model"
        )  # class in charge of storing the data
        track_mate_settings = sj.jimport(
            "fiji.plugin.trackmate.Settings"
        )  # class storing the fields that will configure TrackMate and pilot how the data is created
        track_mate = sj.jimport("fiji.plugin.trackmate.TrackMate")
        logger = sj.jimport("fiji.plugin.trackmate.Logger")
        lap_utils = sj.jimport(
            "fiji.plugin.trackmate.tracking.jaqaman.LAPUtils"
        )
        sparse_lap_tracker_factory = sj.jimport(
            "fiji.plugin.trackmate.tracking.jaqaman.SparseLAPTrackerFactory"
        )
        file = sj.jimport("java.io.File")
        tm_xml_writer = sj.jimport("fiji.plugin.trackmate.io.TmXmlWriter")
        tracker_keys = sj.jimport("fiji.plugin.trackmate.tracking.TrackerKeys")
        track_mate_track_analyzer_provider = sj.jimport(
            "fiji.plugin.trackmate.providers.TrackAnalyzerProvider"
        )
        track_mate_edge_analyzer_provider = sj.jimport(
            "fiji.plugin.trackmate.providers.EdgeAnalyzerProvider"
        )

        if fast_mode:
            try:
                cellpose_detector_factory = sj.jimport(
                    "fiji.plugin.trackmate.cellpose.tbonte.CellposeDetectorFactory"
                )
                pretrained_model = sj.jimport(
                    "fiji.plugin.trackmate.cellpose.tbonte.CellposeSettings.PretrainedModel"
                )
            except TypeError:
                print(
                    "No Trackmate plugin found for fast mode. Using usual segmentation instead."
                )
                fast_mode = False

        if not fast_mode:
            cellpose_detector_factory = sj.jimport(
                "fiji.plugin.trackmate.cellpose.CellposeDetectorFactory"
            )
            pretrained_model = sj.jimport(
                "fiji.plugin.trackmate.cellpose.CellposeSettings.PretrainedModel"
            )

        # Skip if file already exists
        video_file_name = os.path.basename(video_path).split(".")[0]
        out_file = os.path.join(save_folder, f"{video_file_name}_model.xml")
        if os.path.exists(out_file):
            print(f"File {out_file} already exists. Skipping.")
            return

        # Get currently selected image
        options = importer_options()
        options.setColorMode(importer_options.COLOR_MODE_GRAYSCALE)
        options.setId(video_path)
        imps = bf_plugin.openImagePlus(options)
        imp = imps[0]

        # Swap Z and T dimensions if necessary
        dims = imp.getDimensions()
        if imp.dims[-1] == "Z":
            imp.setDimensions(
                sj.to_java(dims[2]), sj.to_java(dims[4]), sj.to_java(dims[3])
            )
            print("Swapping Z and T dimensions")

        # Get the average spot size to define max linking distances
        average_spot_size = models.CellposeModel(
            pretrained_model=[self.model_path]
        ).diam_labels

        # Create model object
        model = track_mate_model()

        # Send all messages to ImageJ log window.
        model.setLogger(logger.IJ_LOGGER)

        # Prepare settings object
        settings = track_mate_settings(imp)

        # Configure detector - We use the Strings for the keys
        settings.detectorFactory = cellpose_detector_factory()
        settings.detectorSettings["TARGET_CHANNEL"] = sj.to_java(3)
        settings.detectorSettings["OPTIONAL_CHANNEL_2"] = sj.to_java(0)
        settings.detectorSettings["CELLPOSE_PYTHON_FILEPATH"] = sys.executable
        settings.detectorSettings["CELLPOSE_MODEL_FILEPATH"] = self.model_path
        settings.detectorSettings["CELLPOSE_MODEL"] = pretrained_model.CUSTOM
        settings.detectorSettings["CELL_DIAMETER"] = sj.to_java(
            0.0, type="double"
        )
        settings.detectorSettings["USE_GPU"] = (
            True if torch.cuda.is_available() else False
        )
        settings.detectorSettings["SIMPLIFY_CONTOURS"] = True

        if fast_mode:
            settings.detectorSettings["FLOW_THRESHOLD"] = self.flow_threshold
            settings.detectorSettings["CELLPROB_THRESHOLD"] = (
                self.cellprob_threshold
            )
            settings.detectorSettings["AUGMENT"] = self.augment

        # Configure tracker
        settings.trackerFactory = sparse_lap_tracker_factory()
        settings.trackerSettings = (
            lap_utils.getDefaultSegmentSettingsMap()
        )  # almost good enough
        settings.trackerSettings["LINKING_MAX_DISTANCE"] = (
            self.linking_max_distance_ratio * average_spot_size
        )
        settings.trackerSettings["ALLOW_GAP_CLOSING"] = True
        settings.trackerSettings["GAP_CLOSING_MAX_DISTANCE"] = (
            self.gap_closing_max_distance_ratio * average_spot_size
        )
        settings.trackerSettings["MAX_FRAME_GAP"] = sj.to_java(
            self.max_frame_gap
        )
        settings.trackerSettings["ALLOW_TRACK_MERGING"] = False
        settings.trackerSettings["ALLOW_TRACK_SPLITTING"] = False
        settings.trackerSettings["LINKING_FEATURE_PENALTIES"] = (
            tracker_keys.DEFAULT_LINKING_FEATURE_PENALTIES
        )

        settings.initialSpotFilterValue = -1.0

        # Add useful track analyzers
        track_analyzer_provider = track_mate_track_analyzer_provider()
        settings.addTrackAnalyzer(
            track_analyzer_provider.getFactory("Track duration")
        )
        settings.addTrackAnalyzer(
            track_analyzer_provider.getFactory("Track index")
        )

        edge_analyzer_provider = track_mate_edge_analyzer_provider()
        settings.addEdgeAnalyzer(
            edge_analyzer_provider.getFactory("Edge target")
        )

        # Instantiate plugin
        trackmate = track_mate(model, settings)

        # Process
        process_ok = trackmate.checkInput()
        if not process_ok:
            sys.exit(str(trackmate.getErrorMessage()))

        process_ok = trackmate.process()
        if not process_ok:
            sys.exit(str(trackmate.getErrorMessage()))

        # Echo results with the logger we set at start:
        model.getLogger().log(str(model))

        out_file_model = file(save_folder, f"{video_file_name}_model.xml")
        writer = tm_xml_writer(out_file_model)
        writer.appendModel(model)
        writer.appendSettings(settings)
        writer.writeToFile()

        # Force exit
        ij_instance.dispose()

        return [], []

    @staticmethod
    def get_spots_from_cellpose(
        cellpose_results: np.ndarray,
    ) -> dict[int, list[CellSpot]]:
        """
        Extract spots from cellpose results.

        Parameters:
            cellpose_results (np.ndarray): TYX
        """

        cell_dictionary: dict[int, list[CellSpot]] = {}
        for frame in range(len(cellpose_results)):
            L = []
            X, Y, cellsframe = barycenter(cellpose_results, frame)
            for id_number in range(1, len(cellsframe) + 1):
                cell_id = cellsframe[id_number - 1]
                cell_coords = []
                x, y = X[id_number - 1], Y[id_number - 1]
                for i in range(len(cell_id[0])):
                    cell_coords.append([cell_id[1][i], cell_id[0][i]])
                cell_coords = np.array(cell_coords)
                hull = ConvexHull(cell_coords)
                convex_hull_indices = cell_coords[hull.vertices][
                    :, ::-1
                ]  # (x, y)
                spot_points = convex_hull_indices
                abs_min_x, abs_max_x, abs_min_y, abs_max_y = (
                    np.abs(np.min(cell_id[1])),
                    np.abs(np.max(cell_id[1])),
                    np.abs(np.min(cell_id[0])),
                    np.abs(np.max(cell_id[0])),
                )
                cell_spot = CellSpot(
                    frame,
                    x,
                    y,
                    id_number,
                    abs_min_x,
                    abs_max_x,
                    abs_min_y,
                    abs_max_y,
                    spot_points,
                )
                L.append(cell_spot)
            cell_dictionary[frame] = L

        return cell_dictionary

    def perform_segmentation_tracking(
        self,
        video: np.ndarray,
    ) -> tuple[list[CellSpot], list[CellTrack]]:
        """

        Parameters:
            video (np.ndarray): TXYC
        """

        # Cellpose segmentation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.CellposeModel(
            pretrained_model=[self.model_path], device=device
        )

        # Reorder video dimension from TXYC to TCYX
        video = np.transpose(video, (0, 3, 2, 1))

        # Expect TCYX
        cellpose_results, _, _ = model.eval(  # TYX
            video,
            channels=[3, 0],
            diameter=0,
            flow_threshold=self.flow_threshold,
            cellprob_threshold=self.cellprob_threshold,
            augment=self.augment,
            resample=False,
        )

        cell_spots_dictionary = self.get_spots_from_cellpose(cellpose_results)

        tracking_method = SpatialLapTrack(
            spatial_coord_slice=slice(0, 2),
            spatial_metric="euclidean",
            track_dist_metric="euclidean",
            track_cost_cutoff=85.5,
            gap_closing_dist_metric="euclidean",
            gap_closing_cost_cutoff=85.5,
            gap_closing_max_frame_count=3,
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
