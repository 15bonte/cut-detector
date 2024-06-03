import os
import concurrent.futures
from typing import Literal, Optional, Callable, Union

import numpy as np
from bigfish import stack, detection
from skimage.morphology import extrema, opening
from scipy import ndimage
from shapely.ops import nearest_points
from shapely import Polygon, Point
from cnn_framework.utils.display_tools import display_progress

from ..utils.cell_track import CellTrack
from ..constants.tracking import CYTOKINESIS_DURATION
from ..utils.mid_body_track import MidBodyTrack
from ..utils.image_tools import smart_cropping
from ..utils.mid_body_spot import MidBodySpot
from ..utils.mitosis_track import MitosisTrack
from ..utils.trackmate_track import TrackMateTrack
from ..utils.factory_plot_detection import plot_detection
from ..utils.gen_track import generate_tracks_from_spots, TRACKING_METHOD
from ..utils.mid_body_track_color_manager import MbTrackColorManager
from ..utils.cell_spot import CellSpot

from ..utils.mb_support import detection as mbd
from ..utils.mb_support import tracking as mbt


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

        cytokinesis_duration (int): Number of frames to look for mid-body between cells.
        minimum_mid_body_track_length (int): Minimum spots in mid-body track to consider it.
    """

    def __init__(
        self,
        track_linking_max_distance=175,
        h_maxima_threshold=5.0,
        sigma=2.0,
        threshold=1.0,
        cytokinesis_duration=CYTOKINESIS_DURATION,
        minimum_mid_body_track_length=5,
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
        ],
    ]

    def update_mid_body_spots(
        self,
        mitosis_track: MitosisTrack,
        mitosis_movie: np.ndarray,
        mask_movie: np.ndarray,
        tracks: list[TrackMateTrack],
        parallel_detection: bool,
        mb_detect_method: SPOT_DETECTION_METHOD = mbd.cur_dog,
        mb_tracking_method: TRACKING_METHOD = mbt.cur_spatial_laptrack,
        log_blob_spot: bool = False,
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
            parallelization=parallel_detection,
            mode=mb_detect_method,
            log_blob_spot=log_blob_spot,
            mitosis_track=mitosis_track,
        )

        mid_body_tracks: list[MidBodyTrack] = generate_tracks_from_spots(
            spots_candidates, mb_tracking_method
        )

        kept_track = self._select_best_track(
            mitosis_track,
            mid_body_tracks,
            tracks,
            mitosis_movie[..., 0],  # sir-tubulin channel
            self.track_linking_max_distance,
        )

        if kept_track is None:
            # If necessary, remove mid_body_spots from mitosis_track
            mitosis_track.mid_body_spots = {}
            return

        # Interpolate mid-body spots if gaps in track
        kept_track.fill_gaps()

        # Save spots of best mitosis track
        for rel_frame, spot in kept_track.spots.items():
            frame = rel_frame + mitosis_track.min_frame
            mitosis_track.mid_body_spots[frame] = spot

    def detect_mid_body_spots(
        self,
        mitosis_movie: np.ndarray,
        mask_movie: Optional[np.ndarray] = None,
        parallelization: bool = False,
        mid_body_channel=1,
        sir_channel=0,
        mode: SPOT_DETECTION_METHOD = mbd.cur_dog,
        log_blob_spot: bool = False,
        mitosis_track: Optional[MitosisTrack] = None,
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

        assert isinstance(
            parallelization, bool
        ), "non-bool parallelization has been deprecated"
        if parallelization:
            return self.thread_pool_detect_mid_body_spots(
                mitosis_movie,
                mask_movie,
                mid_body_channel,
                sir_channel,
                mode,
                mitosis_track,
            )
        else:
            return self.serial_detect_mid_body_spots(
                mitosis_movie,
                mask_movie,
                mid_body_channel,
                sir_channel,
                mode,
                log_blob_spot,
                mitosis_track,
            )

    def serial_detect_mid_body_spots(
        self,
        mitosis_movie: np.ndarray,
        mask_movie: np.ndarray,
        mid_body_channel=1,
        sir_channel=0,
        mode: SPOT_DETECTION_METHOD = mbd.cur_dog,
        log_blob_spot: bool = False,
        mitosis_track: Optional[MitosisTrack] = None,
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
                mitosis_track,
            )

            # Update dictionary
            spots_dictionary[frame] = spots

        return spots_dictionary

    def thread_pool_detect_mid_body_spots(
        self,
        mitosis_movie: np.array,
        mask_movie: np.array,
        mid_body_channel=1,
        sir_channel=0,
        method: SPOT_DETECTION_METHOD = mbd.cur_log,
        mitosis_track: Optional[MitosisTrack] = None,
    ) -> dict[int, list[MidBodySpot]]:

        nb_frames = mitosis_movie.shape[0]

        framed_sd = lambda i, m, mbc, sc, d, f: (
            f,
            self._spot_detection(i, m, mbc, sc, d, f, False, mitosis_track),
        )

        future_list = []
        with concurrent.futures.ThreadPoolExecutor() as e:
            for f in range(nb_frames):
                future_list.append(
                    e.submit(
                        framed_sd,
                        mitosis_movie[f],
                        mask_movie[f],
                        mid_body_channel,
                        sir_channel,
                        method,
                        f,
                    )
                )

        return {
            res.result()[0]: res.result()[1]
            for res in concurrent.futures.as_completed(future_list)
        }

    def _spot_detection(
        self,
        image: np.array,
        mask: np.array,
        mid_body_channel: int,
        sir_channel: int,
        mode: SPOT_DETECTION_METHOD,
        frame: int,
        log_blob_spot: bool = False,
        mitosis_track: Optional[MitosisTrack] = None,
    ) -> list[MidBodySpot]:
        """
        Mode 'bigfish'
            threshold_1: sigma for log filter
            threshold_2: threshold for spots detection

        Mode 'h_maxima'
            threshold_1: threshold for h_maxima
            threshold_2: unused
        """

        if mitosis_track is None:
            image_sir = image[:, :, sir_channel]
            image_mklp = image[:, :, mid_body_channel]
            shift_x, shift_y = 0, 0
        elif isinstance(mitosis_track, MitosisTrack):
            mitosis_position = (
                mitosis_track.position
            )  # mitosis position in movie
            frame_position = mitosis_track.dln_positions[
                frame + mitosis_track.min_frame
            ]  # frame position in mitosis
            shift_x = frame_position.min_x - mitosis_position.min_x
            shift_y = frame_position.min_y - mitosis_position.min_y
            image_sir = image[
                shift_y : frame_position.max_y - mitosis_position.min_y,
                shift_x : frame_position.max_x - mitosis_position.min_x,
                sir_channel,
            ]
            image_mklp = image[
                shift_y : frame_position.max_y - mitosis_position.min_y,
                shift_x : frame_position.max_x - mitosis_position.min_x,
                mid_body_channel,
            ]
        else:
            raise RuntimeError(
                f"Invalid type for arg mitosis_track: {mitosis_track}"
            )

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
                "cur_log": mbd.cur_log,
                "cur_dog": mbd.cur_dog,
                "cur_doh": mbd.cur_doh,
                "lapgau": mbd.lapgau,
                "log2_wider": mbd.log2_wider,
                "rshift_log": mbd.rshift_log,
                "diffgau": mbd.diffgau,
                "hessian": mbd.hessian,
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
                x=position[1] + shift_x,
                y=position[0] + shift_y,
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

    def get_mid_body_expected_positions(
        self,
        mitosis_track: MitosisTrack,
        cell_tracks: list[CellTrack],
    ) -> tuple[dict, dict[int, CellSpot], dict[int, CellSpot]]:
        """Compute Mid-body expected positions for first cytokinesis frames.
        Defined at the point where the two cells are the closest.
        Outputs are relative to the mitosis_track position.
        """
        (
            mother_track,
            daughter_tracks,
        ) = mitosis_track.get_mother_daughters_tracks(cell_tracks)
        # NB: only first daughter is considered
        daughter_track = daughter_tracks[0]

        rel_expected_positions = {}
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

            # Get relative positions
            rel_positions_mother = [
                [
                    int(mother_point[0]) - mitosis_track.position.min_x,
                    int(mother_point[1]) - mitosis_track.position.min_y,
                ]
                for mother_point in mother_track.spots[frame].spot_points
            ]
            rel_positions_daughter = [
                [
                    int(daughter_point[0]) - mitosis_track.position.min_x,
                    int(daughter_point[1]) - mitosis_track.position.min_y,
                ]
                for daughter_point in daughter_track.spots[frame].spot_points
            ]

            # Try to get points in both cells
            mid_body_candidates = [
                position
                for position in rel_positions_mother
                if position in rel_positions_daughter
            ]
            if (
                len(mid_body_candidates) > 0
            ):  # get mean of mid_body_candidates (as before)
                mid_body_position = np.mean(mid_body_candidates, axis=0)
            else:  # else, get mean of closest_points
                mother_polygon = Polygon(rel_positions_mother)
                daughter_polygon = Polygon(rel_positions_daughter)
                nearest = nearest_points(mother_polygon, daughter_polygon)
                mid_body_point = Point(
                    (nearest[0].x + nearest[1].x) / 2,
                    (nearest[0].y + nearest[1].y) / 2,
                )
                mid_body_position = np.array(
                    [mid_body_point.x, mid_body_point.y]
                )

            rel_expected_positions[frame - mitosis_track.min_frame] = (
                mid_body_position
            )

        return (
            rel_expected_positions,
            mother_track.spots,
            daughter_track.spots,
        )

    def _select_best_track(
        self,
        mitosis_track: MitosisTrack,
        mid_body_tracks: list[MidBodyTrack],
        trackmate_tracks: list[TrackMateTrack],
        tubulin_movie: np.ndarray,
        mid_body_linking_max_distance: float,
    ) -> MidBodyTrack:
        """
        Select best track from mid-body tracks.

        Parameters
        ----------
        tubulin_movie: TYX
            relative frames
        """

        # Filter: keep only tracks with sir-tubulin signal at cytokinesis
        # Define range of frames to study
        abs_min_frame = mitosis_track.key_events_frame[
            "cytokinesis"
        ]  # Cytokinesis start
        abs_max_frame = abs_min_frame + int(self.cytokinesis_duration / 2)
        # Randomly sample 1000 intensities to compute threshold
        intensities = np.random.choice(
            tubulin_movie[
                abs_min_frame
                - mitosis_track.min_frame : abs_max_frame
                - mitosis_track.min_frame
                + 1,
                ...,
            ].flatten(),
            1000,
        )
        threshold = np.percentile(intensities, 75)
        # Iterate over all tracks and keep only those with high sir-tubulin signal
        kept_tracks: list[MidBodyTrack] = []
        for track in mid_body_tracks:
            abs_track_frames = [
                frame + mitosis_track.min_frame
                for frame in list(track.spots.keys())
            ]
            # Ignore if no frame in common
            if (
                abs_min_frame > abs_track_frames[-1]
                or abs_max_frame < abs_track_frames[0]
            ):
                continue
            frame_count, total_sir_intensity = 0, 0
            for abs_frame in range(abs_min_frame, abs_max_frame):
                if abs_frame not in abs_track_frames:
                    continue
                frame_count += 1
                track_spot = track.spots[abs_frame - mitosis_track.min_frame]
                total_sir_intensity += tubulin_movie[
                    abs_frame - mitosis_track.min_frame,
                    track_spot.y,
                    track_spot.x,
                ]
            # Ignore if mid-body is not detected in enough frames
            if frame_count < self.minimum_mid_body_track_length:
                continue
            # Ignore if sir-tubulin signal is not high enough
            if total_sir_intensity / frame_count < threshold:
                continue
            kept_tracks.append(track)

        # If no track has sir-tubulin signal, return None
        if len(kept_tracks) == 0:
            return None

        # Sort: choose closest to mid-body expected position
        expected_positions, _, _ = self.get_mid_body_expected_positions(
            mitosis_track, trackmate_tracks
        )
        # Get list of expected distances
        expected_distances = []
        for track in kept_tracks:
            expected_distances.append(
                track.get_expected_distance(
                    expected_positions,
                    mid_body_linking_max_distance,
                )
            )
        assert len(expected_distances) == len(kept_tracks)

        # Sort tracks by expected distance
        sorted_tracks = sorted(
            kept_tracks,
            key=lambda track: expected_distances[kept_tracks.index(track)],
        )
        return sorted_tracks[0]

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
