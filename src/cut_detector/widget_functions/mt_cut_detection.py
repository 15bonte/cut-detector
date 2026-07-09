import os
import pickle
from typing import Optional
import numpy as np

from ..utils.parameters import Parameters
from ..factories.mt_cut_detection_factory import MtCutDetectionFactory
from ..models.tools import get_model_path
from ..utils.mitosis_track import MitosisTrack


def perform_mt_cut_detection(
    raw_video: np.ndarray,
    video_name: str,
    exported_mitoses_dir: str,
    params: Parameters,
    hmm_bridges_parameters_file: Optional[str] = None,
    bridges_mt_cnn_model_path: Optional[str] = None,
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
    params: Parameters
        Video parameters.
    hmm_bridges_parameters_file : Optional[str], optional
        HMM bridges parameters file.
    bridges_mt_cnn_model_path : Optional[str], optional
        Bridges micro-tubules CNN model path.
    save : bool, optional
        Save updated mitosis tracks, by default True.
    """
    if hmm_bridges_parameters_file is None:
        hmm_bridges_parameters_file = os.path.join(
            get_model_path("hmm"), "hmm_bridges_parameters.npz"
        )
    if bridges_mt_cnn_model_path is None:
        bridges_mt_cnn_model_path = get_model_path("bridges_mt_cnn")

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
    mt_cut_detector = MtCutDetectionFactory(params)
    mt_cut_detector.update_mt_cut_detection(
        mitosis_tracks,
        raw_video,
        hmm_bridges_parameters_file,
        bridges_mt_cnn_model_path,
    )

    # Save updated mitosis tracks
    if save:
        for mitosis_track in mitosis_tracks:
            state_path = f"{mitosis_track.get_file_name(video_name)}.bin"
            save_path = os.path.join(
                exported_mitoses_dir,
                state_path,
            )
            with open(save_path, "wb") as f:
                pickle.dump(mitosis_track, f)

    return mitosis_tracks
