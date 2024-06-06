import os
import pickle
from typing import Optional, Callable, Union
import numpy as np
from aicsimageio.writers import OmeTiffWriter
from laptrack import LapTrack
from tqdm import tqdm

from ..factories.mid_body_detection_factory import MidBodyDetectionFactory

from ..utils.mb_support import detection, tracking
from ..utils.mitosis_track import MitosisTrack
from ..utils.cell_track import CellTrack


def perform_mid_body_detection(
    raw_video: np.ndarray,
    video_name: str,
    exported_mitoses_dir: str,
    exported_tracks_dir: str,
    movies_save_dir: Optional[str] = None,
    save: bool = True,
    mid_body_detection_method: Union[
        str, Callable[[np.ndarray], np.ndarray]
    ] = detection.cur_log,
    mid_body_tracking_method: Union[
        str, LapTrack
    ] = tracking.cur_spatial_laptrack,
    parallel_detection: bool = False,
    target_mitosis_id: Optional[int] = None,
) -> list[MitosisTrack]:
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

    # Load cell tracks
    cell_tracks: list[CellTrack] = []
    # Iterate over "bin" files in exported_tracks_dir
    video_exported_tracks_dir = os.path.join(exported_tracks_dir, video_name)
    for state_path in os.listdir(video_exported_tracks_dir):
        # Load mitosis track
        with open(
            os.path.join(video_exported_tracks_dir, state_path), "rb"
        ) as f:
            cell_track: CellTrack = pickle.load(f)
            cell_track.adapt_deprecated_attributes()
            cell_tracks.append(cell_track)

    print("### MID-BODY DETECTION ###")

    # Generate movie for each mitosis and save
    mid_body_detector = MidBodyDetectionFactory()
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
            mb_detect_method=mid_body_detection_method,
            mb_tracking_method=mid_body_tracking_method,
        )

        # Save updated mitosis track
        if save:
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

        if movies_save_dir:
            # Save mitosis movie
            final_mitosis_movie = mitosis_track.add_mid_body_movie(
                mitosis_movie, mask_movie
            )  # TYX C=C+1
            image_save_path = os.path.join(
                movies_save_dir,
                f"{video_name}_mitosis_{mitosis_track.id}_{mitosis_track.mother_track_id}_to_{daughter_track_ids}.tiff",
            )
            # Transpose to match TCYX
            final_mitosis_movie = np.transpose(
                final_mitosis_movie, (0, 3, 1, 2)
            )
            OmeTiffWriter.save(
                final_mitosis_movie, image_save_path, dim_order="TCYX"
            )

    return mitosis_tracks
