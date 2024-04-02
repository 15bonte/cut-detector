import os
import sys
import torch
import imagej
import scyjava as sj
from cellpose import models

from ..utils.cell_track import CellTrack


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
    ) -> None:
        self.model_path = model_path
        self.augment = augment
        self.cellprob_threshold = cellprob_threshold
        self.flow_threshold = flow_threshold
        self.gap_closing_max_distance_ratio = gap_closing_max_distance_ratio
        self.linking_max_distance_ratio = linking_max_distance_ratio
        self.max_frame_gap = max_frame_gap

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
