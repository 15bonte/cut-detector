import os
import pickle
from typing import Optional
import numpy as np

from ..factories.mt_cut_detection_factory import MtCutDetectionFactory
from ..models.tools import get_model_path
from ..utils.mitosis_track import MitosisTrack


def perform_mt_cut_detection(
    raw_video: np.ndarray,
    video_name: str,
    exported_mitoses_dir: str,
    scaler_path: Optional[str] = os.path.join(
        get_model_path("svc_bridges"), "scaler.pkl"
    ),
    model_path: Optional[str] = os.path.join(
        get_model_path("svc_bridges"), "model.pkl"
    ),
    hmm_bridges_parameters_file: Optional[str] = os.path.join(
        get_model_path("hmm"), "hmm_bridges_parameters.npz"
    ),
    bridges_mt_cnn_model_path: Optional[str] = get_model_path(
        "bridges_mt_cnn"
    ),
    update_mitoses: Optional[bool] = True,
):
    """

    Parameters
    ----------
    raw_video: np.ndarray
        TYXC

    """
    mitosis_tracks: list[MitosisTrack] = []
    # Iterate over "bin" files in exported_mitoses_dir
    for state_path in os.listdir(exported_mitoses_dir):
        # Ignore if not for current video
        if video_name not in state_path:
            continue
        # Load mitosis track
        with open(os.path.join(exported_mitoses_dir, state_path), "rb") as f:
            mitosis_track: MitosisTrack = pickle.load(f)
            mitosis_track.adapt_deprecated_attributes()

        # Add mitosis track to list
        mitosis_tracks.append(mitosis_track)

    # Generate a list of the first MT cut time for each mitosis track
    mt_cut_detector = MtCutDetectionFactory()
    list_class_bridges = mt_cut_detector.classify_bridges(
        mitosis_tracks, raw_video, bridges_mt_cnn_model_path
    )

    for i, mitosis_track in enumerate(mitosis_tracks):
        print(f"Detect MT cut ({i+1}/{len(mitosis_tracks)})...")

        # Perform cut detection
        mt_cut_detector.update_mt_cut_detection(
            mitosis_track,
            raw_video,
            scaler_path,
            model_path,
            hmm_bridges_parameters_file,
            list_class_bridges=list_class_bridges[mitosis_track.id],
        )

        # Save updated mitosis track
        if update_mitoses:
            daughter_track_ids = ",".join(
                [str(d) for d in mitosis_track.daughter_track_ids]
            )
            state_path = f"{video_name}_mitosis_{mitosis_track.id}_{mitosis_track.mother_track_id}_to_{daughter_track_ids}.bin"
            save_path = os.path.join(
                exported_mitoses_dir,
                state_path,
            )
            with open(save_path, "wb") as f:
                pickle.dump(mitosis_track, f)
