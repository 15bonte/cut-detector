import os
import concurrent.futures
from typing import Literal, Optional, Callable, Union

import numpy as np
from bigfish import stack, detection
from skimage.morphology import extrema, opening
from scipy import ndimage

from cnn_framework.utils.display_tools import display_progress

from ..constants.tracking import CYTOKINESIS_DURATION
from ..utils.mid_body_track import MidBodyTrack
from ..utils.image_tools import smart_cropping
from ..utils.mid_body_spot import MidBodySpot
from ..utils.mitosis_track import MitosisTrack
from ..utils.trackmate_track import TrackMateTrack
from ..utils.factory_plot_detection import plot_detection
from ..utils.gen_track import generate_tracks_from_spots, TRACKING_METHOD
from ..utils.mid_body_track_color_manager import MbTrackColorManager

from ..utils.mb_support import detection as mbd
from ..utils.mb_support import tracking  as mbt

class MidBodyDetectionFactory:
    """
    Class to perform mid-body detection, tracking and filtering.

    Args:
        weight_mklp_intensity_factor (float): Weight of intensity in spot dist calculation
            (cf TrackMate).
        weight_sir_intensity_factor (float): Weight of sir intensity in spot distance calculation.
        mid_body_linking_max_distance (int): Maximum distance between two mid-bodies to link them.

        h_maxima_threshold (float): Threshold for h_maxima detection (default).

        sigma (float): Sigma for bigfish detection (unused).
        threshold (float): Threshold for bigfish detection (unused).

        cytokinesis_duration (int): Number of frames to look for mid-body in between cells.
        minimum_mid_body_track_length (int): Minimum spots in mid-body track to consider it.
    """

    def __init__(
        self,
        track_linking_max_distance=175,
        h_maxima_threshold=5.0,
        sigma=2.0,
        threshold=1.0,
        cytokinesis_duration=CYTOKINESIS_DURATION,
        minimum_mid_body_track_length=10,
    ) -> None:
        self.track_linking_max_distance = track_linking_max_distance
        self.h_maxima_threshold = h_maxima_threshold
        self.sigma = sigma
        self.threshold = threshold
        self.cytokinesis_duration = cytokinesis_duration
        self.minimum_mid_body_track_length = minimum_mid_body_track_length

    SPOT_DETECTION_METHOD = Union[
        Callable[[np.ndarray], np.ndarray],
        Literal[
            "bigfish",
            "h_maxima",

            "cur_log",
            "lapgau",
            "log2_wider",
            "rshift_log",

            "cur_dog",
            "diffgau",

            "cur_doh",
            "hessian",
        ]
    ]

    def update_mid_body_spots(
        self,
        mitosis_track: MitosisTrack,
        mitosis_movie: np.ndarray,
        mask_movie:    np.ndarray,
        tracks:        list[TrackMateTrack],
        mb_detect_method:   SPOT_DETECTION_METHOD = mbd.cur_dog,
        mb_tracking_method: TRACKING_METHOD       = mbt.cur_spatial_laptrack,
        log_blob_spot:      bool = False,
        parallel_detection: bool = False,
    ) -> None:
        """
        Get spots of best mitosis track.

        Parameters
        ----------
        mitosis_movie: TYXC
        mask_movie: TYX

        """

        spots_candidates = self.detect_mid_body_spots(
            mitosis_movie,
            mask_movie=mask_movie,
            mode=mb_detect_method,
            log_blob_spot=log_blob_spot,
            parallelization=parallel_detection
        )

        mid_body_tracks: list[MidBodyTrack] = generate_tracks_from_spots(
            spots_candidates,
            mb_tracking_method
        )

        kept_track = self._select_best_track(
            mitosis_track,
            mid_body_tracks,
            tracks,
            mitosis_movie,
            self.track_linking_max_distance,
        )

        if kept_track is None:
            return

        # Keep only spots of best mitosis track
        for rel_frame, spot in kept_track.spots.items():
            frame = rel_frame + mitosis_track.min_frame
            mitosis_track.mid_body_spots[frame] = spot

    def detect_mid_body_spots(
            self,
            mitosis_movie: np.ndarray,
            mask_movie: Optional[np.ndarray] = None,
            mid_body_channel=1,
            sir_channel=0,
            mode: SPOT_DETECTION_METHOD = mbd.cur_dog,
            log_blob_spot: bool = False,
            parallelization: bool = False,
            ) -> dict[int, list[MidBodySpot]]:
        """
        Parameters
        ----------
        mitosis_movie: TYXC
        mask_movie: TYX

        Returns
        ----------
        spots_dictionary: dict[int, list[MidBodySpot]]
        """

        # Default mask is all ones
        if mask_movie is None:
            mask_movie = np.ones(mitosis_movie.shape[:-1])

        assert isinstance(parallelization, bool), "non-bool parallelization has been deprecated"
        if parallelization:
            return self.thread_pool_detect_mid_body_spots(
                mitosis_movie,
                mask_movie,
                mid_body_channel,
                sir_channel,
                mode
            )
        else:
            return self.serial_detect_mid_body_spots(
                mitosis_movie,
                mask_movie,
                mid_body_channel,
                sir_channel,
                mode,
                log_blob_spot
            )


    def serial_detect_mid_body_spots(
            self,
            mitosis_movie: np.ndarray,
            mask_movie:    np.ndarray,
            mid_body_channel = 1,
            sir_channel      = 0,
            mode: SPOT_DETECTION_METHOD = mbd.cur_dog,
            log_blob_spot: bool = False,
            ) -> dict[int, list[MidBodySpot]]:

        spots_dictionary = {}
        nb_frames = mitosis_movie.shape[0]

        for frame in range(nb_frames):
            display_progress(
                "Detect mid-body spots...",
                frame + 1,
                nb_frames,
                additional_message=f"Frame {frame + 1}/{nb_frames}",
            )

            mitosis_frame = mitosis_movie[frame]  # YXC
            mask_frame = mask_movie[frame]  # YX
            spots = self._spot_detection(
                mitosis_frame,
                mask_frame,
                mid_body_channel,
                sir_channel,
                mode,
                frame,
                log_blob_spot,
            )

            # Update dictionary
            spots_dictionary[frame] = spots

        return spots_dictionary

    def thread_pool_detect_mid_body_spots(
            self,
            mitosis_movie: np.array,
            mask_movie:    np.array,
            mid_body_channel = 1,
            sir_channel      = 0,
            method: SPOT_DETECTION_METHOD = mbd.cur_log
            ) -> dict[int, list[MidBodySpot]]:

        nb_frames = mitosis_movie.shape[0]

        framed_sd = lambda i, m, mbc, sc, d, f: (f, self._spot_detection(i, m, mbc, sc, d, f, False))

        future_list = []
        with concurrent.futures.ThreadPoolExecutor() as e:
            for f in range(nb_frames):
                future_list.append(e.submit(
                    framed_sd,
                    mitosis_movie[f],
                    mask_movie,
                    mid_body_channel,
                    sir_channel,
                    method,
                    f,
                ))

        return {
            res.result()[0]: res.result()[1]
            for res in concurrent.futures.as_completed(future_list)
        }

    def _spot_detection(
        self,
        image: np.array,
        mask:  np.array,
        mid_body_channel: int,
        sir_channel:      int,
        mode: SPOT_DETECTION_METHOD,
        frame: int,
        log_blob_spot: bool = False,
    ) -> list[MidBodySpot]:
        """
        Mode 'bigfish'
            threshold_1: sigma for log filter
            threshold_2: threshold for spots detection

        Mode 'h_maxima'
            threshold_1: threshold for h_maxima
            threshold_2: unused
        """

        image_sir = image[:, :, sir_channel]
        image_mklp = image[:, :, mid_body_channel]  #

        if callable(mode):
            # directly passsing a blob-like function
            spots = [
                (int(spot[0]), int(spot[1]), int(spot[2]))
                for spot in mode(image_mklp)
            ]
            if log_blob_spot:
                for s in spots:
                    print(f"found x:{s[1]}  y:{s[0]}  s:{s[2]}")

        elif mode in [
            "cur_log",
            "lapgau",
            "log2_wider",
            "rshift_log",
            "cur_dog",
            "diffgau",
            "cur_doh",
            "hessian",
        ]:
            # blob-like function called referenced by name

            mapping = {
                "cur_log":    mbd.cur_log,
                "cur_dog":    mbd.cur_dog,
                "cur_doh":    mbd.cur_doh,
                "lapgau":     mbd.lapgau,
                "log2_wider": mbd.log2_wider,
                "rshift_log":  mbd.rshift_log,
                "diffgau":    mbd.diffgau,
                "hessian":    mbd.hessian,
            }

            spots = [
                (int(spot[0]), int(spot[1]), int(spot[2]))
                for spot in mapping[mode](image_mklp)
            ]

            if log_blob_spot:
                for s in spots:
                    print(f"found x:{s[1]}  y:{s[0]}  s:{s[2]}")

        elif mode == "bigfish":
            # Spots detection with bigfish functions
            filtered_image = stack.log_filter(image_mklp, sigma=self.sigma)
            # Filter out spots which are not maximal or outside convex hull
            spots_mask = (filtered_image > 0) * mask
            # If mask is empty, skip frame
            if np.sum(spots_mask) == 0:
                spots = np.array([], dtype=np.int64).reshape(
                    (0, filtered_image.ndim)
                )
            else:
                spots, _ = detection.spots_thresholding(
                    filtered_image,
                    spots_mask.astype(bool),
                    threshold=self.threshold,
                )

        elif mode == "h_maxima":
            # Perform opening followed by closing to remove small spots
            filtered_image = opening(image_mklp, footprint=np.ones((3, 3)))
            # Get local maxima using h_maxima
            local_maxima = extrema.h_maxima(
                filtered_image, self.h_maxima_threshold
            )
            # Label spot regions
            labeled_local_maxima, nb_labels = ndimage.label(
                local_maxima, structure=np.ones((3, 3))
            )
            # Remove inconsistent labels
            # Threshold is computed as 99th percentile of image
            filtering_threshold = np.quantile(image_mklp.flatten(), 0.99)
            for label in range(1, nb_labels + 1):
                # Labels intensity in original image has to be higher than threshold
                if (
                    image_mklp[np.where(labeled_local_maxima == label)].mean()
                    < filtering_threshold
                ):
                    labeled_local_maxima[labeled_local_maxima == label] = 0
            # Re-label accordingly
            labeled_local_maxima, nb_labels = ndimage.label(
                labeled_local_maxima > 0, structure=np.ones((3, 3))
            )
            # Get center of mass to locate spots
            spots = ndimage.center_of_mass(
                local_maxima, labeled_local_maxima, range(1, nb_labels + 1)
            )
            spots = np.asarray(spots, dtype=np.int64)

            # Here, do something to retrieve mid_body area and/or circularity...
            if len(spots) == 0:
                spots = np.array([], dtype=np.int64).reshape(
                    (0, filtered_image.ndim)
                )

        else:
            raise ValueError(f"Unknown mode: [{mode}]")

        # WARNING:
        # spots can be a list of Tuple with 2 or 3 values:
        # 2 values: (y, x) if h_maxima or fish_eye used
        # 3 values: (y, x, sigma) if any blob-based method used
        mid_body_spots = [
            MidBodySpot(
                frame,
                # Convert spots to MidBodySpot objects (switch (y, x) to (x, y))
                x=position[1],
                y=position[0],
                intensity=self._get_average_intensity(position, image_mklp),
                sir_intensity=self._get_average_intensity(position, image_sir),
            )
            for position in spots
        ]
        return mid_body_spots

    @staticmethod
    def _get_average_intensity(
        position: tuple[int], image: np.array, margin=1
    ) -> int:
        """
        Parameters
        ----------
        position: (y, x)
        image: YX
        margin: int

        Returns
        ----------
        average_intensity: int
        """
        # Get associated crop
        crop = smart_cropping(
            image,
            margin,
            position[1],
            position[0],
            position[1] + 1,
            position[0] + 1,
        )

        # Return average intensity
        return int(np.mean(crop))

    def _select_best_track(
        self,
        mitosis_track: MitosisTrack,
        mid_body_tracks: list[MidBodyTrack],
        trackmate_tracks: list[TrackMateTrack],
        mitosis_movie: np.ndarray,
        mid_body_linking_max_distance: float,
        sir_channel=0,
    ) -> MidBodyTrack:
        """
        Select best track from mid-body tracks.

        Parameters
        ----------
        mitosis_movie: TYXC
        """
        (
            mother_track,
            daughter_tracks,
        ) = mitosis_track.get_mother_daughters_tracks(trackmate_tracks)
        # NB: only first daughter is considered
        daughter_track = daughter_tracks[0]

        expected_positions = {}
        for frame in range(
            daughter_track.start,
            daughter_track.start + self.cytokinesis_duration,
        ):
            # If one cell does not exist anymore, stop
            if (
                frame not in daughter_track.spots
                or frame not in mother_track.spots
            ):
                continue
            # Compute mid-body expected relative position at current frame
            closest_points = []
            min_distance = np.inf
            for mother_point in mother_track.spots[frame].spot_points:
                position_mother = [
                    int(mother_point[0]) - mitosis_track.position.min_x,
                    int(mother_point[1]) - mitosis_track.position.min_y,
                ]
                for daughter_point in daughter_track.spots[frame].spot_points:
                    position_daughter = [
                        int(daughter_point[0]) - mitosis_track.position.min_x,
                        int(daughter_point[1]) - mitosis_track.position.min_y,
                    ]
                    distance = np.linalg.norm(
                        [
                            a - b
                            for a, b in zip(position_mother, position_daughter)
                        ]
                    )
                    if distance < min_distance:
                        min_distance = distance
                        closest_points = [(position_mother, position_daughter)]
                    if distance == min_distance:
                        closest_points.append(
                            (position_mother, position_daughter)
                        )

            mid_body_position = np.mean(closest_points, axis=0)
            mid_body_position = np.mean(mid_body_position, axis=0)
            expected_positions[frame - mitosis_track.min_frame] = (
                mid_body_position
            )

        # Remove wrong tracks by keeping only tracks with at least minimum_track_length points
        mid_body_tracks = [
            track
            for track in mid_body_tracks
            if track.length > self.minimum_mid_body_track_length
        ]

        # Compute mean intensity on sir-tubulin channel for each track
        image_sir = mitosis_movie[..., sir_channel]  # TYX
        sir_intensity_track = [0 for _ in mid_body_tracks]
        for idx, track in enumerate(mid_body_tracks):
            abs_track_frames = [
                frame + mitosis_track.min_frame
                for frame in list(track.spots.keys())
            ]
            abs_min_frame = mitosis_track.key_events_frame[
                "cytokinesis"
            ]  # Cytokinesis start
            abs_max_frame = abs_min_frame + int(self.cytokinesis_duration / 2)
            if (
                abs_min_frame > abs_track_frames[-1]
                or abs_max_frame < abs_track_frames[0]
            ):
                sir_intensity_track[idx] = -np.inf
            frame_count = 0
            for frame in range(abs_min_frame, abs_max_frame + 1):
                if frame not in abs_track_frames:
                    continue
                frame_count += 1
                track_spot = track.spots[frame - mitosis_track.min_frame]
                sir_intensity_track[idx] += image_sir[
                    frame - mitosis_track.min_frame,
                    track_spot.y,
                    track_spot.x,
                ]

            if frame_count < (abs_max_frame - abs_min_frame + 1) / 2:
                sir_intensity_track[idx] = -np.inf
            else:
                sir_intensity_track[idx] /= frame_count

        # Get list of expected distances
        expected_distances = []
        for track in mid_body_tracks:
            val = track.get_expected_distance(
                expected_positions, mid_body_linking_max_distance
            )
            expected_distances.append(val)

        # Assert lists have same length for next function
        assert len(expected_distances) == len(mid_body_tracks)
        assert len(sir_intensity_track) == len(mid_body_tracks)

        # function to sort tracks by expected distance and intensity
        def func_sir_intensity(track):
            a = expected_distances[mid_body_tracks.index(track)]
            b = sir_intensity_track[mid_body_tracks.index(track)]
            return a - 0.5 * b

        # Remove tracks with infinite func value
        fun_values = []
        final_tracks = []
        for track in mid_body_tracks:
            fun_values.append(func_sir_intensity(track))
            if func_sir_intensity(track) != np.inf:
                final_tracks.append(track)

        # Sort tracks by func value
        sorted_tracks = sorted(final_tracks, key=func_sir_intensity)
        return sorted_tracks[0] if len(sorted_tracks) > 0 else None

    def save_mid_body_tracking(
        self,
        spots_candidates,
        mitosis_movie: np.ndarray,
        path_output: str,
        mid_body_channel=1,
    ):
        """
        Plot spots detection & tracking.
        """
        # Check if directory exists
        if not os.path.exists(path_output):
            os.makedirs(path_output)

        color_lib = MbTrackColorManager()

        # Detect spots in each frame
        nb_frames = mitosis_movie.shape[0]
        for frame in range(nb_frames):
            print(f"generating and saving frame ({frame+1}/{nb_frames})")
            image = mitosis_movie[frame, :, :, mid_body_channel].squeeze()

            # Bigfish spots
            frame_spots = [
                [spot.y, spot.x] for spot in spots_candidates[frame]
            ]
            colors = [
                (
                    # matplotlib_colors[spot.track_id % len(matplotlib_colors)]
                    color_lib.get_color_for_track(spot.track_id)
                    if spot.track_id is not None
                    else (0, 0, 0)
                )
                for spot in spots_candidates[frame]
            ]

            plot_detection(
                image,
                frame_spots,
                color=colors,
                contrast=True,
                path_output=os.path.join(
                    path_output, f"spot_detection_{frame}.png"
                ),
                show=False,
                title=f"Python frame {frame} - Fiji frame {frame + 1}",
                fill=True,
            )
