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
    hmm_bridges_parameters_file: Optional[str] = os.path.join(
        get_model_path("hmm"), "hmm_bridges_parameters.npz"
    ),
    bridges_mt_cnn_model_path: Optional[str] = get_model_path(
        "bridges_mt_cnn"
    ),
    save: bool = True,
) -> list[MitosisTrack]:
    """Perform micro-tubules cut detection.

    Parameters
    ----------
    raw_video : np.ndarray
        Raw video. TYXC.
    video_name : str
        Video name.
    exported_mitoses_dir : str
        Directory where mitosis tracks are saved.
    hmm_bridges_parameters_file : Optional[str], optional
        HMM bridges parameters file.
    bridges_mt_cnn_model_path : Optional[str], optional
        Bridges micro-tubules CNN model path.
    save : bool, optional
        Save updated mitosis tracks, by default True.
    """
    print("\n### MICRO-TUBULES CUT DETECTION ###")

    mitosis_tracks: list[MitosisTrack] = []
    # Iterate over "bin" files in exported_mitoses_dir
    for state_path in os.listdir(exported_mitoses_dir):
        # Ignore if not for current video
        if video_name not in state_path:
            continue
        # Load mitosis track
        with open(os.path.join(exported_mitoses_dir, state_path), "rb") as f:
            mitosis_track = MitosisTrack.load(f)

        # Add mitosis track to list
        mitosis_tracks.append(mitosis_track)

    # Perform cut detection
    mt_cut_detector = MtCutDetectionFactory()
    mt_cut_detector.update_mt_cut_detection(
        mitosis_tracks,
        raw_video,
        hmm_bridges_parameters_file,
        bridges_mt_cnn_model_path,
    )

    # Save updated mitosis tracks
    if save:
        for mitosis_track in mitosis_tracks:
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

    return mitosis_tracks
