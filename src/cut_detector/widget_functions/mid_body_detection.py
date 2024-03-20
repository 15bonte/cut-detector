import os
import pickle
from typing import Optional
import numpy as np
from aicsimageio.writers import OmeTiffWriter

from ..factories.mid_body_detection_factory import MidBodyDetectionFactory

from ..utils.mitosis_track import MitosisTrack
from ..utils.trackmate_track import TrackMateTrack


def perform_mid_body_detection(
    raw_video: np.ndarray,
    video_name: str,
    exported_mitoses_dir: str,
    exported_tracks_dir: str,
    save_dir: Optional[str] = None,
    update_mitoses: bool = True,
):
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

    # Load trackmate tracks
    trackmate_tracks: list[TrackMateTrack] = []
    # Iterate over "bin" files in exported_tracks_dir
    video_exported_tracks_dir = os.path.join(exported_tracks_dir, video_name)
    for state_path in os.listdir(video_exported_tracks_dir):
        # Load mitosis track
        with open(
            os.path.join(video_exported_tracks_dir, state_path), "rb"
        ) as f:
            trackmate_track: TrackMateTrack = pickle.load(f)
            trackmate_track.adapt_deprecated_attributes()
            trackmate_tracks.append(trackmate_track)

    # Generate movie for each mitosis and save
    mid_body_detector = MidBodyDetectionFactory()
    for i, mitosis_track in enumerate(mitosis_tracks):
        print(f"\nGenerate mitosis movies ({i+1}/{len(mitosis_tracks)})...")

        # Generate mitosis movie
        mitosis_movie, mask_movie = mitosis_track.generate_video_movie(
            raw_video
        )  # TYXC, TYX

        # Search for mid-body in mitosis movie
        mid_body_detector.update_mid_body_spots(
            mitosis_track, mitosis_movie, mask_movie, trackmate_tracks
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

        if save_dir:
            # Save mitosis movie
            final_mitosis_movie = mitosis_track.add_mid_body_movie(
                mitosis_movie, mask_movie
            )  # TYX C=C+1
            image_save_path = os.path.join(
                save_dir,
                f"{video_name}_mitosis_{mitosis_track.id}_{mitosis_track.mother_track_id}_to_{daughter_track_ids}.tiff",
            )
            # Transpose to match TCYX
            final_mitosis_movie = np.transpose(
                final_mitosis_movie, (0, 3, 1, 2)
            )
            OmeTiffWriter.save(
                final_mitosis_movie, image_save_path, dim_order="TCYX"
            )
