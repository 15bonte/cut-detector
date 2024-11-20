import os
import pickle
from typing import Optional
import numpy as np
from aicsimageio.writers import OmeTiffWriter
from tqdm import tqdm


from ..factories.mid_body_detection_factory import MidBodyDetectionFactory

from ..utils.mitosis_track import MitosisTrack
from ..utils.cell_track import CellTrack
from ..utils.parameters import Parameters


def perform_mid_body_detection(
    raw_video: np.ndarray,
    video_name: str,
    exported_mitoses_dir: str,
    exported_tracks_dir: str,
    movies_save_dir: Optional[str] = None,
    save: bool = True,
    parallel_detection: bool = False,
    detection_method: str = "difference_gaussian",
    target_mitosis_id: Optional[int] = None,
    params=Parameters(),
) -> list[MitosisTrack]:
    """Perform mid-body detection on mitosis tracks.

    Parameters
    ----------
    raw_video : np.ndarray
        Raw video to extract mitosis movies from. TYXC.
    video_name : str
        Name of the video.
    exported_mitoses_dir : str
        Directory where mitosis tracks are saved.
    exported_tracks_dir : str
        Directory where cell tracks are saved.
    movies_save_dir : Optional[str], optional
        Directory where mitosis movies are saved, by default None.
    save : bool, optional
        Save updated mitosis tracks, by default True.
    parallel_detection : bool, optional
        Perform detection in parallel, by default False.
    detection_method : str, optional
        Detection method to use, by default "difference_gaussian".
    target_mitosis_id : Optional[int], optional
        Target mitosis id to perform mid-body detection on, by default None.
    params : Parameters, optional
        Video parameters.

    Returns
    -------
    list[MitosisTrack]
        List of updated mitosis tracks.
    """
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

    # Load cell tracks
    cell_tracks: list[CellTrack] = []
    # Iterate over "bin" files in exported_tracks_dir
    video_exported_tracks_dir = os.path.join(exported_tracks_dir, video_name)
    for state_path in os.listdir(video_exported_tracks_dir):
        # Load mitosis track
        with open(
            os.path.join(video_exported_tracks_dir, state_path), "rb"
        ) as f:
            cell_track = CellTrack.load(f)
            cell_tracks.append(cell_track)

    print("\n### MID-BODY DETECTION ###")

    # Generate movie for each mitosis and save
    print("Performing mid-body detection.")
    mid_body_detector = MidBodyDetectionFactory(params)
    for i, mitosis_track in enumerate(tqdm(mitosis_tracks)):

        if (
            isinstance(target_mitosis_id, int)
            and mitosis_track.id != target_mitosis_id
        ):
            print(
                f"\nTrack {i+1}/{len(mitosis_tracks)}, Mitosis id {mitosis_track.id} - Skipped"
            )
            continue

        # Generate mitosis movie
        mitosis_movie, mask_movie = mitosis_track.generate_video_movie(
            raw_video
        )  # TYXC, TYX

        # Search for mid-body in mitosis movie
        mid_body_detector.update_mid_body_spots(
            mitosis_track,
            mitosis_movie,
            cell_tracks,
            parallel_detection=parallel_detection,
            detection_method=detection_method,
        )

        # Save updated mitosis track
        if save:
            state_path = f"{mitosis_track.get_file_name(video_name)}.bin"
            save_path = os.path.join(
                exported_mitoses_dir,
                state_path,
            )
            with open(save_path, "wb") as f:
                pickle.dump(mitosis_track, f)

        if movies_save_dir:
            # Save mitosis movie
            final_mitosis_movie = mitosis_track.add_mid_body_movie(
                mitosis_movie, mask_movie
            )  # TYX C=C+1
            image_save_path = os.path.join(
                movies_save_dir,
                f"{mitosis_track.get_file_name(video_name)}.tiff",
            )
            # Transpose to match TCYX
            final_mitosis_movie = np.transpose(
                final_mitosis_movie, (0, 3, 1, 2)
            )
            OmeTiffWriter.save(
                final_mitosis_movie, image_save_path, dim_order="TCYX"
            )

    return mitosis_tracks
