import os
import sys
from typing import Optional
import torch
import imagej
import scyjava as sj
from cellpose import models

from ..models.tools import get_model_path
from ..constants.tracking import (
    AUGMENT,
    CELLPROB_THRESHOLD,
    FLOW_THRESHOLD,
    GAP_CLOSING_MAX_DISTANCE_RATIO,
    LINKING_MAX_DISTANCE_RATIO,
    MAX_FRAME_GAP,
)


def perform_tracking(
    video_path: str,
    fiji_path: str,
    save_folder: str,
    model_path: Optional[str],
    fast_mode: bool,
) -> None:
    # Load default model if necessary
    if model_path is None:
        model_path = get_model_path("segmentation_model")

    # Create save directory if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

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
    lap_utils = sj.jimport("fiji.plugin.trackmate.tracking.jaqaman.LAPUtils")
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
            print("No Trackmate plugin found for fast mode. Using usual segmentation instead.")
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
        imp.setDimensions(sj.to_java(dims[2]), sj.to_java(dims[4]), sj.to_java(dims[3]))
        print("Swapping Z and T dimensions")

    # Get the average spot size to define max linking distances
    average_spot_size = models.CellposeModel(pretrained_model=[model_path]).diam_labels

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
    settings.detectorSettings["CELLPOSE_MODEL_FILEPATH"] = model_path
    settings.detectorSettings["CELLPOSE_MODEL"] = pretrained_model.CUSTOM
    settings.detectorSettings["CELL_DIAMETER"] = sj.to_java(0.0, type="double")
    settings.detectorSettings["USE_GPU"] = True if torch.cuda.is_available() else False
    settings.detectorSettings["SIMPLIFY_CONTOURS"] = True

    if fast_mode:
        settings.detectorSettings["FLOW_THRESHOLD"] = FLOW_THRESHOLD
        settings.detectorSettings["CELLPROB_THRESHOLD"] = CELLPROB_THRESHOLD
        settings.detectorSettings["AUGMENT"] = AUGMENT

    # Configure tracker
    settings.trackerFactory = sparse_lap_tracker_factory()
    settings.trackerSettings = lap_utils.getDefaultSegmentSettingsMap()  # almost good enough
    settings.trackerSettings["LINKING_MAX_DISTANCE"] = (
        LINKING_MAX_DISTANCE_RATIO * average_spot_size
    )
    settings.trackerSettings["ALLOW_GAP_CLOSING"] = True
    settings.trackerSettings["GAP_CLOSING_MAX_DISTANCE"] = (
        GAP_CLOSING_MAX_DISTANCE_RATIO * average_spot_size
    )
    settings.trackerSettings["MAX_FRAME_GAP"] = sj.to_java(MAX_FRAME_GAP)
    settings.trackerSettings["ALLOW_TRACK_MERGING"] = False
    settings.trackerSettings["ALLOW_TRACK_SPLITTING"] = False
    settings.trackerSettings[
        "LINKING_FEATURE_PENALTIES"
    ] = tracker_keys.DEFAULT_LINKING_FEATURE_PENALTIES

    settings.initialSpotFilterValue = -1.0

    # Add useful track analyzers
    track_analyzer_provider = track_mate_track_analyzer_provider()
    settings.addTrackAnalyzer(track_analyzer_provider.getFactory("Track duration"))
    settings.addTrackAnalyzer(track_analyzer_provider.getFactory("Track index"))

    edge_analyzer_provider = track_mate_edge_analyzer_provider()
    settings.addEdgeAnalyzer(edge_analyzer_provider.getFactory("Edge target"))

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

    print("Process finished with success!")

    # Force exit
    ij_instance.dispose()
