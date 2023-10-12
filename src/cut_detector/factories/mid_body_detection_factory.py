import os
from typing import Optional
import numpy as np
from bigfish import stack, detection, plot
from skimage.morphology import extrema, opening
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from cnn_framework.utils.display_tools import display_progress

from ..constants.tracking import CYTOKINESIS_DURATION
from ..utils.mid_body_track import MidBodyTrack
from ..utils.image_tools import smart_cropping
from ..utils.mid_body_spot import MidBodySpot
from ..utils.mitosis_track import MitosisTrack
from ..utils.trackmate_track import TrackMateTrack


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
        weight_mklp_intensity_factor=10.0,
        weight_sir_intensity_factor=3.33,
        mid_body_linking_max_distance=100,
        h_maxima_threshold=5.0,
        sigma=2.0,
        threshold=1.0,
        cytokinesis_duration=CYTOKINESIS_DURATION,
        minimum_mid_body_track_length=10,
    ) -> None:
        self.weight_mklp_intensity_factor = weight_mklp_intensity_factor
        self.weight_sir_intensity_factor = weight_sir_intensity_factor
        self.mid_body_linking_max_distance = mid_body_linking_max_distance
        self.h_maxima_threshold = h_maxima_threshold
        self.sigma = sigma
        self.threshold = threshold
        self.cytokinesis_duration = cytokinesis_duration
        self.minimum_mid_body_track_length = minimum_mid_body_track_length

    def update_mid_body_spots(
        self,
        mitosis_track: MitosisTrack,
        mitosis_movie: np.array,
        mask_movie: np.array,
        tracks: list[TrackMateTrack],
        path_output: Optional[str] = None,
    ) -> None:
        """
        Get spots of best mitosis track.

        Parameters
        ----------
        mitosis_movie: (T, H, W, C)
        mask_movie: (T, H, W)

        """

        spots_candidates = self._detect_mid_body_spots(
            mitosis_movie, mask_movie=mask_movie, path_output=path_output
        )
        mid_body_tracks = self._generate_tracks_from_spots(spots_candidates)
        kept_track = self._select_best_track(mitosis_track, mid_body_tracks, tracks, mitosis_movie)

        if kept_track is None:
            return

        # Keep only spots of best mitosis track
        for rel_frame, spot in kept_track.spots.items():
            frame = rel_frame + mitosis_track.min_frame
            mitosis_track.mid_body_spots[frame] = spot

    def _detect_mid_body_spots(
        self,
        mitosis_movie: np.array,
        mask_movie: Optional[np.array] = None,
        mid_body_channel=1,
        sir_channel=0,
        path_output: Optional[str] = None,
        mode="h_maxima",
    ) -> dict[int, list[MidBodySpot]]:
        """
        Parameters
        ----------
        mitosis_movie: (T, H, W, C)
        mask_movie: (T, H, W)

        Returns
        ----------
        spots_dictionary: dict[int, list[MidBodySpot]]
        """

        # Default mask is all ones
        if mask_movie is None:
            mask_movie = np.ones(mitosis_movie.shape[:-1])

        # Detect spots in each frame
        spots_dictionary = {}
        nb_frames = mitosis_movie.shape[0]
        for frame in range(nb_frames):
            display_progress(
                "Detect mid-body spots...",
                frame + 1,
                nb_frames,
                additional_message=f"Frame {frame + 1}/{nb_frames}",
            )

            mitosis_frame = mitosis_movie[frame, :, :, :].squeeze()  # H, W, C
            mask_frame = mask_movie[frame, :, :].squeeze()  # H, W
            spots = self.spot_detection(
                mitosis_frame,
                mask_frame,
                mid_body_channel,
                sir_channel,
                path_output=path_output,
                mode=mode,
                frame=frame,
            )

            # Update dictionary
            spots_dictionary[frame] = spots

        return spots_dictionary

    def spot_detection(
        self,
        image: np.array,
        mask: np.array,
        mid_body_channel: int,
        sir_channel: int,
        path_output: Optional[str] = None,
        mode="bigfish",
        frame=-1,
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

        if mode == "bigfish":
            # Spots detection with bigfish functions
            filtered_image = stack.log_filter(image_mklp, sigma=self.sigma)
            filtered_image = image_mklp
            # Filter out spots which are not maximal or outside convex hull
            spots_mask = (filtered_image > 0) * mask
            # If mask is empty, skip frame
            if np.sum(spots_mask) == 0:
                spots = np.array([], dtype=np.int64).reshape((0, filtered_image.ndim))
            else:
                spots, _ = detection.spots_thresholding(
                    filtered_image, spots_mask.astype(np.bool), threshold=self.threshold
                )

        elif mode == "h_maxima":
            # Perform opening followed by closing to remove small spots
            filtered_image = opening(image_mklp, footprint=np.ones((3, 3)))
            # Get local maxima using h_maxima
            local_maxima = extrema.h_maxima(filtered_image, self.h_maxima_threshold)
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
                spots = np.array([], dtype=np.int64).reshape((0, filtered_image.ndim))

        else:
            raise ValueError(f"Unknown mode: {mode}")

        if path_output is not None:
            # Check if directory exists
            if not os.path.exists(path_output):
                os.makedirs(path_output)
            # Count number of files in directory
            nb_files = len(os.listdir(path_output))
            plot.plot_detection(
                filtered_image,
                spots,
                contrast=True,
                path_output=os.path.join(path_output, f"spot_detection_{nb_files+1}.png"),
                show=False,
            )

        # Convert spots to MidBodySpot objects (switch (y, x) to (x, y))
        mid_body_spots = [
            MidBodySpot(
                frame,
                (position[1], position[0]),
                intensity=self._get_average_intensity(position, image_mklp),
                sir_intensity=self._get_average_intensity(position, image_sir),
            )
            for position in spots
        ]

        return mid_body_spots

    @staticmethod
    def _get_average_intensity(position: tuple[int], image: np.array, margin=1) -> int:
        """
        Parameters
        ----------
        position: (y, x)
        image: (H, W)
        margin: int

        Returns
        ----------
        average_intensity: int
        """
        # Get associated crop
        crop = smart_cropping(
            image, margin, position[1], position[0], position[1] + 1, position[0] + 1
        )

        # Return average intensity
        return int(np.mean(crop))

    def _update_spots_hereditary(
        self, spots1: list[MidBodySpot], spots2: list[MidBodySpot]
    ) -> None:
        """
        Link spots together using Hungarian algorithm.
        """
        # Ignore empty spots list
        if len(spots1) == 0 or len(spots2) == 0:
            return

        # Create cost matrix
        # https://imagej.net/plugins/trackmate/algorithms
        cost_matrix = np.zeros((len(spots1) + len(spots2), len(spots1) + len(spots2)))
        max_cost = 0
        for i, spot1 in enumerate(spots1):
            for j, spot2 in enumerate(spots2):
                intensity_penalty = (
                    3
                    * self.weight_mklp_intensity_factor
                    * np.abs(spot1.intensity - spot2.intensity)
                    / (spot1.intensity + spot2.intensity)
                )
                sir_intensity_penalty = (
                    3
                    * self.weight_sir_intensity_factor
                    * np.abs(spot1.sir_intensity - spot2.sir_intensity)
                    / (spot1.sir_intensity + spot2.sir_intensity)
                )
                penalty = 1 + intensity_penalty + sir_intensity_penalty
                distance = np.linalg.norm(np.array(spot1.position) - np.array(spot2.position))
                if distance > self.mid_body_linking_max_distance:
                    cost_matrix[i, j] = np.inf
                else:
                    # Compared to original TrackMate algorithm, remove square to penalize no attribution to the closest spot
                    cost_matrix[i, j] = (penalty * distance) ** 1
                    max_cost = max(max_cost, cost_matrix[i, j])

        min_cost = 0 if np.max(cost_matrix) == 0 else np.min(cost_matrix[np.nonzero(cost_matrix)])

        cost_matrix[len(spots1) :, : len(spots2)] = max_cost * 1.05  # bottom left
        cost_matrix[: len(spots1), len(spots2) :] = max_cost * 1.05  # top right
        cost_matrix[len(spots1) :, len(spots2) :] = min_cost  # bottom right

        # Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Update parent and child spots
        for i, j in zip(row_ind, col_ind):
            if i < len(spots1) and j < len(spots2):
                spots1[i].child_spot = spots2[j]
                spots2[j].parent_spot = spots1[i]

    def _generate_tracks_from_spots(
        self, spots_candidates: dict[int, list[MidBodySpot]]
    ) -> list[MidBodyTrack]:
        """
        Use spots linked together to generate tracks.

        Parameters
        ----------
        spots_candidates : {frame: [MidBodySpot]}

        Returns
        ----------
        tracks: [MidBodyTrack]

        """

        # Update parent and child spots
        for frame in spots_candidates.keys():
            # Ignore if no spot detected in next frame
            if not frame + 1 in spots_candidates:
                continue
            self._update_spots_hereditary(spots_candidates[frame], spots_candidates[frame + 1])

        tracks = []
        for spots in spots_candidates.values():
            for spot in spots:
                if spot.track_id is None:
                    # Create new track
                    track_id = len(tracks)
                    new_track = MidBodyTrack(track_id)
                    new_track.add_spot(spot)
                    tracks.append(new_track)
        return tracks

    def _select_best_track(
        self,
        mitosis_track: MitosisTrack,
        mid_body_tracks: list[MidBodyTrack],
        trackmate_tracks: list[TrackMateTrack],
        mitosis_movie: np.array,
        sir_channel=0,
    ) -> MidBodyTrack:
        """
        Select best track from mid-body tracks.
        """
        mother_track, daughter_tracks = mitosis_track.get_mother_daughters_tracks(trackmate_tracks)
        # NB: only first daughter is considered
        daughter_track = daughter_tracks[0]

        expected_positions = {}
        for frame in range(daughter_track.start, daughter_track.start + self.cytokinesis_duration):
            # If one cell does not exist anymore, stop
            if frame not in daughter_track.track_spots or frame not in mother_track.track_spots:
                continue
            # Compute mid-body expected relative position at current frame
            closest_points = []
            min_distance = np.inf
            for mother_point in mother_track.track_spots[frame].spot_points:
                position_mother = [
                    int(mother_point[0]) - mitosis_track.position.min_x,
                    int(mother_point[1]) - mitosis_track.position.min_y,
                ]
                for daughter_point in daughter_track.track_spots[frame].spot_points:
                    position_daughter = [
                        int(daughter_point[0]) - mitosis_track.position.min_x,
                        int(daughter_point[1]) - mitosis_track.position.min_y,
                    ]
                    distance = np.linalg.norm(
                        [a - b for a, b in zip(position_mother, position_daughter)]
                    )
                    if distance < min_distance:
                        min_distance = distance
                        closest_points = [(position_mother, position_daughter)]
                    if distance == min_distance:
                        closest_points.append((position_mother, position_daughter))

            mid_body_position = np.mean(closest_points, axis=0)
            mid_body_position = np.mean(mid_body_position, axis=0)
            expected_positions[frame - mitosis_track.min_frame] = mid_body_position

        # Remove wrong tracks by keeping only tracks with at least minimum_track_length points
        mid_body_tracks = [
            track for track in mid_body_tracks if track.length > self.minimum_mid_body_track_length
        ]

        # Compute mean intensity on sir-tubulin channel for each track
        image_sir = mitosis_movie[:, :, :, sir_channel]
        sir_intensity_track = [0 for _ in mid_body_tracks]
        for idx, track in enumerate(mid_body_tracks):
            abs_track_frames = [
                frame + mitosis_track.min_frame for frame in list(track.spots.keys())
            ]
            abs_min_frame = mitosis_track.key_events_frame["cytokinesis"]  # Cytokinesis start
            abs_max_frame = abs_min_frame + int(self.cytokinesis_duration / 2)
            if abs_min_frame > abs_track_frames[-1] or abs_max_frame < abs_track_frames[0]:
                sir_intensity_track[idx] = -np.inf
            frame_count = 0
            for frame in range(abs_min_frame, abs_max_frame + 1):
                if frame not in abs_track_frames:
                    continue
                frame_count += 1
                track_position = track.spots[frame - mitosis_track.min_frame].position
                sir_intensity_track[idx] += image_sir[
                    frame - mitosis_track.min_frame, track_position[1], track_position[0]
                ]

            if frame_count < (abs_max_frame - abs_min_frame + 1) / 2:
                sir_intensity_track[idx] = -np.inf
            else:
                sir_intensity_track[idx] /= frame_count

        # Get list of expected distances
        expected_distances = []
        for track in mid_body_tracks:
            val = track.get_expected_distance(
                expected_positions, self.mid_body_linking_max_distance
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
