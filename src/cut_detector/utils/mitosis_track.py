from __future__ import annotations

from copy import deepcopy
import json
import os
import pickle
from typing import Optional, Tuple, Union
import numpy as np
from skimage.morphology import extrema, opening
import xmltodict
from munch import Munch
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy.optimize import linear_sum_assignment
from scipy import ndimage
from bigfish import stack, detection, plot
from cnn_framework.utils.display_tools import display_progress

from ..constants.annotations import NAMES_DICTIONARY, MARGIN
from ..constants.tracking import (
    CYTOKINESIS_DURATION,
    FRAMES_AROUND_METAPHASE,
    METAPHASE_INDEX,
    MID_BODY_LINKING_MAX_DISTANCE,
    MINIMUM_MID_BODY_TRACK_LENGTH,
    H_MAXIMA_THRESHOLD,
    MINIMUM_DISTANCE_TO_BORDER,
    WEIGHT_MKLP_INTENSITY_FACTOR,
    WEIGHT_SIR_INTENSITY_FACTOR,
)

from .bridges_classification.bridges_classification_parameters import (
    BridgesClassificationParameters,
)
from .mid_body_spot import MidBodySpot
from .mid_body_track import MidBodyTrack
from .trackmate_track import TrackMateTrack
from .box_dimensions_dln import BoxDimensionsDln
from .box_dimensions import BoxDimensions
from .bridges_classification.tools import get_bridge_class, apply_hmm
from .bridges_classification.impossible_detection import ImpossibleDetection
from .image_tools import resize_image, smart_cropping


def spot_detection(
    image: np.array,
    mask: np.array,
    mid_body_channel: int,
    sir_channel: int,
    threshold_1=2,
    threshold_2=1,
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
        filtered_image = stack.log_filter(image_mklp, sigma=threshold_1)
        filtered_image = image_mklp
        # Filter out spots which are not maximal or outside convex hull
        spots_mask = (filtered_image > 0) * mask
        # If mask is empty, skip frame
        if np.sum(spots_mask) == 0:
            spots = np.array([], dtype=np.int64).reshape((0, filtered_image.ndim))
        else:
            spots, _ = detection.spots_thresholding(
                filtered_image, spots_mask.astype(np.bool), threshold=threshold_2
            )

    elif mode == "h_maxima":
        # Perform opening followed by closing to remove small spots
        filtered_image = opening(image_mklp, footprint=np.ones((3, 3)))
        # Get local maxima using h_maxima
        local_maxima = extrema.h_maxima(filtered_image, threshold_1)
        # Label spot regions
        labeled_local_maxima, nb_labels = ndimage.label(local_maxima, structure=np.ones((3, 3)))
        # Remove inconsistent labels
        # Threshold is computed as 99th percentile of image
        filtering_threshold = np.quantile(image_mklp.flatten(), 0.99)
        for label in range(1, nb_labels + 1):
            # Labels intensity in original image has to be higher than threshold
            if image_mklp[np.where(labeled_local_maxima == label)].mean() < filtering_threshold:
                labeled_local_maxima[labeled_local_maxima == label] = 0
        # Re-label accordingly
        labeled_local_maxima, nb_labels = ndimage.label(
            labeled_local_maxima > 0, structure=np.ones((3, 3))
        )
        # Get center of mass to locate spots
        spots = ndimage.center_of_mass(local_maxima, labeled_local_maxima, range(1, nb_labels + 1))
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
            intensity=get_average_intensity(position, image_mklp),
            sir_intensity=get_average_intensity(position, image_sir),
        )
        for position in spots
    ]

    return mid_body_spots


def get_average_intensity(position: tuple(int, int), image: np.array, margin=1) -> int:
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


def detect_mid_body_spots(
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
        spots = spot_detection(
            mitosis_frame,
            mask_frame,
            mid_body_channel,
            sir_channel,
            threshold_1=H_MAXIMA_THRESHOLD,
            threshold_2=None,
            path_output=path_output,
            mode=mode,
            frame=frame,
        )

        # Update dictionary
        spots_dictionary[frame] = spots

    return spots_dictionary


def update_spots_hereditary(spots1: list[MidBodySpot], spots2: list[MidBodySpot]) -> None:
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
                * WEIGHT_MKLP_INTENSITY_FACTOR
                * np.abs(spot1.intensity - spot2.intensity)
                / (spot1.intensity + spot2.intensity)
            )
            sir_intensity_penalty = (
                3
                * WEIGHT_SIR_INTENSITY_FACTOR
                * np.abs(spot1.sir_intensity - spot2.sir_intensity)
                / (spot1.sir_intensity + spot2.sir_intensity)
            )
            penalty = 1 + intensity_penalty + sir_intensity_penalty
            distance = np.linalg.norm(np.array(spot1.position) - np.array(spot2.position))
            if distance > MID_BODY_LINKING_MAX_DISTANCE:
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


def generate_tracks_from_spots(
    spots_candidates: dict[int, list[MidBodySpot]]
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
        update_spots_hereditary(spots_candidates[frame], spots_candidates[frame + 1])

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


class MitosisTrack:
    """
    A class to store the information of a mitosis track.
    """

    def __init__(self, mother_track_id: int, daughter_track_id: int, metaphase_frame: int):
        # Elementary information
        self.mother_track_id = mother_track_id
        self.daughter_track_ids = [daughter_track_id]
        self.id: Optional[int] = None

        self.metaphase_frame = metaphase_frame

        # Key events: metaphase/cytokinesis/first_mt_cut/second_mt_cut/first_membrane_cut
        self.key_events_frame: dict[str, Union[int, ImpossibleDetection]] = {}
        self.gt_key_events_frame: Optional[dict[str, int]] = None

        # Time
        self.min_frame: Optional[int] = None
        self.max_frame: Optional[int] = None

        # Position
        self.position = BoxDimensions()

        # Delaunay triangulation, by frame
        self.dln_positions: dict[int, BoxDimensionsDln] = {}

        # Used for matching between ground truth and prediction
        self.matched = False

        # Mid body spot indexed by absolute frame
        self.mid_body_spots: dict[int, MidBodySpot] = {}
        self.gt_mid_body_spots: Optional[dict[int, MidBodySpot]] = None

        # Used to know if the track is near the border of the video
        self.is_near_border = False

    def is_same_mitosis(self, mother_track_id: int, metaphase_frame: int) -> bool:
        return self.mother_track_id == mother_track_id and self.metaphase_frame == metaphase_frame

    def add_daughter_track(self, daughter_track_id: int) -> None:
        self.daughter_track_ids.append(daughter_track_id)

    def _get_mother_daughters_tracks(
        self, tracks: list[TrackMateTrack]
    ) -> Tuple[TrackMateTrack, list[TrackMateTrack]]:
        mother_track = [track for track in tracks if track.track_id == self.mother_track_id][0]
        daughter_tracks = [track for track in tracks if track.track_id in self.daughter_track_ids]
        return mother_track, daughter_tracks

    def _add_dln_position(self, frame: int, frame_dimensions: BoxDimensionsDln) -> None:
        self.dln_positions[frame] = deepcopy(frame_dimensions)
        # Update absolute min and max accordingly
        self.position.update_from_box_dimensions(frame_dimensions)

    def update_mitosis_start_end(
        self, trackmate_tracks: list[TrackMateTrack], mitosis_tracks: list[MitosisTrack]
    ) -> None:
        # Get all tracks involved in current mitosis
        mother_track, daughter_tracks = self._get_mother_daughters_tracks(trackmate_tracks)

        # Get min and max frame of current mitosis
        # Min is the metaphase frame minus FRAMES_AROUND_METAPHASE, protected against frames before start of mother track
        min_frame = max(
            mother_track.start,
            self.metaphase_frame - FRAMES_AROUND_METAPHASE,
        )
        # For each daughter track, the end is the end of the track OR the next metaphase event of this track
        max_frame = mother_track.stop
        for track in [mother_track] + daughter_tracks:
            track_end_frame = track.stop
            for track_to_merge_bis in mitosis_tracks:
                if (
                    track_to_merge_bis.mother_track_id == track.track_id
                    and track_to_merge_bis.metaphase_frame
                    > self.metaphase_frame  # other mitosis should be strictly after
                ):
                    track_end_frame = min(track_end_frame, track_to_merge_bis.metaphase_frame)
            max_frame = min(max_frame, track_end_frame)

        # Update mitosis_track
        self.min_frame = min_frame
        self.max_frame = max_frame

    def update_is_near_border(self, raw_video: np.array) -> None:
        """
        Parameters
        ----------
        raw_video: np.array  # (T, H, W, C)
        ----------
        """

        max_height, max_width = raw_video.shape[1], raw_video.shape[2]

        cyto_frame = self.key_events_frame["cytokinesis"]
        last_frame = cyto_frame + CYTOKINESIS_DURATION

        # get mitosis coordinates between cyto_frame and last_frame
        min_dist_to_border = np.inf
        for frame in range(cyto_frame, last_frame + 1):
            if frame not in self.mid_body_spots:
                continue

            # get mid-body coordinates
            x_rel = self.mid_body_spots[frame].position[0]
            y_rel = self.mid_body_spots[frame].position[1]

            x_abs = x_rel + self.position.min_x
            y_abs = y_rel + self.position.min_y
            mid_body_coordinates = (x_abs, y_abs)

            # get distance to border
            min_x = min(mid_body_coordinates[0], max_width - mid_body_coordinates[0])
            min_y = min(mid_body_coordinates[1], max_height - mid_body_coordinates[1])

            min_dist_to_border = min(min_dist_to_border, min_x, min_y)

        self.is_near_border = min_dist_to_border < MINIMUM_DISTANCE_TO_BORDER

    def update_key_events_frame(self, trackmate_tracks: list[TrackMateTrack]) -> None:
        # Get all tracks involved in current mitosis
        mother_track, daughter_tracks = self._get_mother_daughters_tracks(trackmate_tracks)

        # Store first metaphase frame
        for frame in range(self.metaphase_frame, mother_track.start, -1):
            # Some frames may be missing since gap closing is allowed
            if frame not in mother_track.track_spots:
                continue
            if mother_track.track_spots[frame].predicted_phase != METAPHASE_INDEX:
                self.key_events_frame["metaphase"] = frame + 1
                break

        # Store first cytokinesis frame - considered as the first frame of daughter tracks
        self.key_events_frame["cytokinesis"] = min([track.start for track in daughter_tracks])

    def update_mitosis_position_dln(self, trackmate_tracks: list[TrackMateTrack]) -> None:
        """
        Update positions of mitosis for each frame and Delaunay triangulation
        """

        min_frame, max_frame = self.min_frame, self.max_frame
        mother_track, daughter_tracks = self._get_mother_daughters_tracks(trackmate_tracks)

        previous_box_dimensions_dln = None
        for frame in range(min_frame, max_frame + 1):
            box_dimensions_dln = mother_track.compute_dln_from_tracks(
                frame, previous_box_dimensions_dln, additional_tracks=daughter_tracks
            )
            # Store in case next frame is missing
            previous_box_dimensions_dln = box_dimensions_dln
            # Update accordingly
            self._add_dln_position(frame, box_dimensions_dln)

    def generate_video_movie(self, raw_video: np.array) -> Tuple[np.array, np.array]:
        """
        Parameters
        ----------
        raw_video: initial video (T, H, W, C)
        ----------

        Returns
        ----------
        mitosis_movie: mitosis movie (T, H, W, C)
        mask_movie: mask movie (T, H, W, C) (all channels are actually the same)
        ----------
        """

        mitosis_movie, mask_movie = [], []
        for frame in range(self.min_frame, self.max_frame + 1):
            # Get useful data for current frame
            min_x = self.dln_positions[frame].min_x
            max_x = self.dln_positions[frame].max_x
            min_y = self.dln_positions[frame].min_y
            max_y = self.dln_positions[frame].max_y
            dln = self.dln_positions[frame].dln

            # Extract frame image, big enough to keep all spots for current track
            frame_image = raw_video[
                frame,
                self.position.min_y : self.position.max_y,
                self.position.min_x : self.position.max_x,
                :,
            ]  # H, W, C

            # Generate mask with Delaunay triangulation
            current_frame_shape = (max_y - min_y, max_x - min_x)  # current spot
            indices = np.stack(np.indices(current_frame_shape), axis=-1)
            out_idx = np.nonzero(dln.find_simplex(indices) + 1)
            single_channel_mask = np.zeros(current_frame_shape)
            single_channel_mask[out_idx] = 1

            # Construct mask image
            mask_image = np.stack([single_channel_mask] * raw_video.shape[-1], axis=0)  # C, H, W
            mask_image = resize_image(
                mask_image,
                method="zero",
                pad_margin_h=[min_y - self.position.min_y, self.position.max_y - max_y],
                pad_margin_w=[min_x - self.position.min_x, self.position.max_x - max_x],
            )[
                0, ...
            ]  # H, W

            mitosis_movie.append(frame_image)
            mask_movie.append(mask_image)

        mitosis_movie = np.array(mitosis_movie)  # T, H, W, C
        mask_movie = np.array(mask_movie)  # T, H, W

        return mitosis_movie, mask_movie

    def _select_best_track(
        self,
        mid_body_tracks: list[MidBodyTrack],
        trackmate_tracks: list[TrackMateTrack],
        mitosis_movie: np.array,
        sir_channel=0,
    ) -> MidBodyTrack:
        """
        Select best track from mid-body tracks.
        """
        mother_track, daughter_tracks = self._get_mother_daughters_tracks(trackmate_tracks)
        # NB: only first daughter is considered
        daughter_track = daughter_tracks[0]

        expected_positions = {}
        for frame in range(daughter_track.start, daughter_track.start + CYTOKINESIS_DURATION):
            # If one cell does not exist anymore, stop
            if frame not in daughter_track.track_spots or frame not in mother_track.track_spots:
                continue
            # Compute mid-body expected relative position at current frame
            closest_points = []
            min_distance = np.inf
            for mother_point in mother_track.track_spots[frame].spot_points:
                position_mother = [
                    int(mother_point[0]) - self.position.min_x,
                    int(mother_point[1]) - self.position.min_y,
                ]
                for daughter_point in daughter_track.track_spots[frame].spot_points:
                    position_daugther = [
                        int(daughter_point[0]) - self.position.min_x,
                        int(daughter_point[1]) - self.position.min_y,
                    ]
                    distance = np.linalg.norm(
                        [a - b for a, b in zip(position_mother, position_daugther)]
                    )
                    if distance < min_distance:
                        min_distance = distance
                        closest_points = [(position_mother, position_daugther)]
                    if distance == min_distance:
                        closest_points.append((position_mother, position_daugther))

            mid_body_position = np.mean(closest_points, axis=0)
            mid_body_position = np.mean(mid_body_position, axis=0)
            expected_positions[frame - self.min_frame] = mid_body_position

        # Remove wrong tracks by keeping only tracks with at least minimum_track_length points
        mid_body_tracks = [
            track for track in mid_body_tracks if track.length > MINIMUM_MID_BODY_TRACK_LENGTH
        ]

        # Compute mean intensity on sir-tubulin channel for each track
        image_sir = mitosis_movie[:, :, :, sir_channel]
        sir_intensity_track = [0 for _ in mid_body_tracks]
        for idx, track in enumerate(mid_body_tracks):
            abs_track_frames = [frame + self.min_frame for frame in list(track.spots.keys())]
            abs_min_frame = self.key_events_frame["cytokinesis"]  # Cytokinesis start
            abs_max_frame = abs_min_frame + int(CYTOKINESIS_DURATION / 2)
            if abs_min_frame > abs_track_frames[-1] or abs_max_frame < abs_track_frames[0]:
                sir_intensity_track[idx] = -np.inf
            frame_count = 0
            for frame in range(abs_min_frame, abs_max_frame + 1):
                if frame not in abs_track_frames:
                    continue
                frame_count += 1
                track_position = track.spots[frame - self.min_frame].position
                sir_intensity_track[idx] += image_sir[
                    frame - self.min_frame, track_position[1], track_position[0]
                ]

            if frame_count < (abs_max_frame - abs_min_frame + 1) / 2:
                sir_intensity_track[idx] = -np.inf
            else:
                sir_intensity_track[idx] /= frame_count

        # Get list of expected distances
        expected_distances = []
        for track in mid_body_tracks:
            val = track.get_expected_distance(expected_positions)
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

    def update_mid_body_spots(
        self,
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

        spots_candidates = detect_mid_body_spots(
            mitosis_movie, mask_movie=mask_movie, path_output=path_output
        )
        mid_body_tracks = generate_tracks_from_spots(spots_candidates)
        kept_track = self._select_best_track(mid_body_tracks, tracks, mitosis_movie)

        if kept_track is None:
            return

        # Keep only spots of best mitosis track
        for rel_frame, spot in kept_track.spots.items():
            frame = rel_frame + self.min_frame
            self.mid_body_spots[frame] = spot

    def generate_mitosis_summary(self, raw_tracks: list[TrackMateTrack], save_path: str) -> None:
        """
        Unused so far.
        Might be improved with all useful information, saved to csv...
        """
        mitosis_summary = {}

        mother_track, daughter_tracks = self._get_mother_daughters_tracks(raw_tracks)
        daughters_first_frame = min([track.start for track in daughter_tracks])

        for idx, frame in enumerate(range(self.min_frame, self.max_frame + 1)):
            # Extreme case where mother track is not present at beginning of mitosis movie
            if frame not in mother_track.track_spots:
                mitosis_summary[idx + 1] = "interphase"
                continue
            # Telophase defined as first frame after metaphase or daughters first frame
            if frame >= self.metaphase_frame or frame >= daughters_first_frame:
                mitosis_summary[idx + 1] = "telophase"
            # Metaphase according to CNN + HMM prediction
            elif mother_track.track_spots[frame].predicted_phase == METAPHASE_INDEX:
                mitosis_summary[idx + 1] = "metaphase"
            # In other cases, interphase
            else:
                mitosis_summary[idx + 1] = "interphase"

        # Save mitosis summary
        with open(save_path, "w") as f:
            json.dump(mitosis_summary, f)

    def is_possible_match(self, other_track: MitosisTrack) -> bool:
        """
        Check if two tracks are a possible match. Other track is typically a ground truth track.
        Match is possible if there is an overlap between the two tracks,
        and other track starts no earlier/no later than FRAMES_AROUND_METAPHASE around self start.
        """
        if abs(other_track.metaphase_frame - self.metaphase_frame) > FRAMES_AROUND_METAPHASE:
            return False

        return self.position.overlaps(other_track.position)

    def add_mid_body_movie(self, mitosis_movie: np.array, mask_movie: np.array) -> np.array:
        """
        Parameters
        ----------
        mitosis_movie: (T, H, W, C)
        mask_movie: (T, H, W)

        Returns
        ----------
        spots_video: (T, H, W, 1)
        """

        video_shape = mitosis_movie.shape[:3]
        spots_video = np.zeros(video_shape)  # T, H, W

        for absolute_frame, spot in self.mid_body_spots.items():
            # Create 1 circle around spot position
            square_size = 2
            spots_video[
                absolute_frame - self.min_frame,
                spot.position[1] - square_size : spot.position[1] + square_size,
                spot.position[0] - square_size : spot.position[0] + square_size,
            ] = 1

        # Add empty dimension at end
        mid_body_movie = np.expand_dims(spots_video, axis=-1)  # (T, H, W, 1)

        # Mix mid-body and mask movie
        mask_movie = np.expand_dims(mask_movie, axis=-1)  # (T, H, W, 1)
        mid_body_mask_movie = mask_movie + mid_body_movie  # (T, H, W, 1)

        # Cast mid_body_mask_movie to mitosis_movie dtype
        mid_body_mask_movie = mid_body_mask_movie.astype(mitosis_movie.dtype)

        mitosis_movie = np.concatenate(
            [mitosis_movie, mid_body_mask_movie], axis=-1
        )  # (T, H, W, C+1)

        return mitosis_movie

    def update_mid_body_ground_truth(self, annotation_file: str, nb_channels: int) -> None:
        """
        Parameters
        ----------
        annotation_file: .xml file with annotations from CellCounter
        nb_channels: number of channels in mitosis movie (very likely to be 4)

        """

        # Initialize gt_key_events_frame - first two events are shared
        self.gt_key_events_frame = {
            "metaphase": self.key_events_frame["metaphase"],
            "cytokinesis": self.key_events_frame["cytokinesis"],
        }
        self.gt_mid_body_spots = {}

        # Read data
        with open(annotation_file) as fd:
            doc = Munch.fromDict(xmltodict.parse(fd.read()))

        for i, type_data in enumerate(doc.CellCounter_Marker_File.Marker_Data.Marker_Type):
            assert i == int(type_data.Type) - 1  # order must be kept
            # Ignore if no data
            if "Marker" not in type_data:
                continue
            markers = type_data.Marker
            if not isinstance(markers, list):
                markers = [markers]
            # Sort markers by frame
            markers = sorted(markers, key=lambda x: int(x.MarkerZ))
            for marker in markers:
                x_pos, y_pos, frame = (
                    int(marker.MarkerX),
                    int(marker.MarkerY),
                    int(marker.MarkerZ) // nb_channels,
                )
                # Create associated spot
                self.gt_mid_body_spots[frame + self.min_frame] = MidBodySpot(frame, (x_pos, y_pos))

            # If Name is missing of wrong, assume it is i
            if "Name" not in type_data or type_data.Name not in NAMES_DICTIONARY:
                class_index = i
            else:
                class_index = NAMES_DICTIONARY[type_data.Name]
                assert class_index == i
            class_first_frame = int(markers[0].MarkerZ) // nb_channels
            class_abs_first_frame = class_first_frame + self.min_frame

            # First MT cut
            if 2 not in self.gt_key_events_frame and class_index % 5 > 0:
                assert (
                    class_abs_first_frame >= self.gt_key_events_frame["cytokinesis"]
                )  # after metaphase
                self.gt_key_events_frame["first_mt_cut"] = class_abs_first_frame

            # Second MT cut
            if 3 not in self.gt_key_events_frame and class_index in [2, 4, 7, 9]:
                assert (
                    class_abs_first_frame >= self.gt_key_events_frame["first_mt_cut"]
                )  # after first MT cut
                self.gt_key_events_frame["second_mt_cut"] = class_abs_first_frame

            # First membrane cut
            if 4 not in self.gt_key_events_frame and class_index in [3, 4, 8, 9]:
                if class_abs_first_frame < self.gt_key_events_frame["first_mt_cut"]:
                    a = 0
                assert (
                    class_abs_first_frame >= self.gt_key_events_frame["first_mt_cut"]
                )  # after first MT cut
                self.gt_key_events_frame["first_membrane_cut"] = class_abs_first_frame

    def evaluate_mid_body_detection(self, tolerance=10, percent_seen=0.9) -> bool:
        """
        Mid_body is considered as detected if during at least percent_seen % of frames
        between cytokinesis and second MT cut it is at most tolerance pixels away
        from ground truth during this interval.

        Parameters
        ----------
        tolerance: maximum distance between ground truth and prediction to consider a match
        percent_seen: minimum percentage of frames where mid_body is seen to consider a match
        """

        position_difference = []

        # Check frames until second MT cut or end of annotations
        max_frame = (
            self.gt_key_events_frame["second_mt_cut"]
            if 3 in self.gt_key_events_frame
            else max(self.gt_mid_body_spots.keys())
        )

        for frame in range(self.gt_key_events_frame["cytokinesis"], max_frame):
            if frame not in self.gt_mid_body_spots:
                continue
            if frame not in self.mid_body_spots.keys():
                position_difference.append(1e3)  # random huge value
                continue
            position_difference.append(
                np.linalg.norm(
                    np.array(self.gt_mid_body_spots[frame].position)
                    - np.array(self.mid_body_spots[frame].position)
                )
            )

        # Get percent_seen th percentile of position difference
        position_difference = np.array(position_difference)
        max_position_difference = np.quantile(position_difference, percent_seen)

        return max_position_difference < tolerance

    def light_spot_detected(
        self,
        video: np.ndarray,
        first_cut_frame: int,
        parameters: BridgesClassificationParameters = BridgesClassificationParameters(),
        print_enabled=False,
    ) -> bool:
        """
        Check if there is a light spot in crops of size crop_size_light_spot
        around the mid-body, in length_light_spot frames around the first micro-tubules cut.

        Spots are detected using h-maxima method with h=h_maxima_light_spot.
        Ignore spots with intensity lower than intensity_threshold_light_spot.
        Ignore spots close to the center (potential mid-bodies), i.e. within
        center_tolerance_light_spot pixels.

        Light spot is considered as detected if at least in min_percentage_light_spot % of frames.
        """
        # Get the mitosis video crop
        video = video[
            :,
            self.position.min_y : self.position.max_y,
            self.position.min_x : self.position.max_x,
            :,
        ]

        nb_spot_detected = 0
        frame_counted = 0
        # Iterate over video frames
        for i in range(-parameters.length_light_spot // 2, parameters.length_light_spot // 2):
            frame = first_cut_frame + i

            # Make sure mid-body exists at frame
            if (
                not self.mid_body_spots
                or frame > self.max_frame
                or frame > max(list(self.mid_body_spots.keys()))
            ):
                continue

            # Get mid-body coordinates
            x_pos, y_pos = self.mid_body_spots[frame].position

            # Extract image and crop on the midbody
            img = np.transpose(video[frame, ...], (2, 0, 1))  # C, H, W
            crop = smart_cropping(img, parameters.crop_size_light_spot, x_pos, y_pos, pad=True)[
                0, ...
            ]  # H, W

            # Perform opening to remove small spots and apply h_maxima to get potential spots
            filtered_image = opening(crop, footprint=np.ones((3, 3)))
            local_maxima = extrema.h_maxima(filtered_image, parameters.h_maxima_light_spot)

            # Label spot regions and remove inconsistent ones
            labeled_local_maxima, nb_labels = ndimage.label(
                local_maxima, structure=np.ones((3, 3))
            )
            for label in range(1, nb_labels + 1):
                # Labels intensity in original image has to be higher than threshold
                if (
                    crop[np.where(labeled_local_maxima == label)].mean()
                    < parameters.intensity_threshold_light_spot
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

            # Remove spots that are too close to the center
            for spot in spots:
                if (
                    np.abs(spot[0] - parameters.crop_size_light_spot)
                    < parameters.center_tolerance_light_spot
                ) and (
                    np.abs(spot[1] - parameters.crop_size_light_spot)
                    < parameters.center_tolerance_light_spot
                ):
                    spots = np.delete(spots, np.where((spots == spot).all(axis=1))[0], axis=0)

            if len(spots) > 0:
                nb_spot_detected += 1
            frame_counted += 1

        # Light spot is considered as detected if at least in MIN_PERCENTAGE_LIGHTSPOT % of frames
        if frame_counted > 0:
            percentage_spot_detected = nb_spot_detected / frame_counted
            spot_detected = percentage_spot_detected >= parameters.min_percentage_light_spot
        else:
            spot_detected = False

        if spot_detected and print_enabled:
            print("nb_spot_detected: ", nb_spot_detected)
            print("frame_counted: ", frame_counted)
            print("spot_detected: ", spot_detected)
            print(
                f"Track: {self.id}_{self.mother_track_id}_to_{','.join(str(daughter) for daughter in self.daughter_track_ids)}"
            )

        return spot_detected

    def is_bridges_classification_impossible(self) -> bool:
        """
        Bridges classification is impossible if:
        - no midbody spots
        - more than 2 daughter tracks
        - nucleus is near border
        - no midbody spot after cytokinesis
        """

        if self.is_near_border:
            self.key_events_frame["first_mt_cut"] = ImpossibleDetection.NEAR_BORDER
            self.key_events_frame["second_mt_cut"] = ImpossibleDetection.NEAR_BORDER
            return True

        if len(self.daughter_track_ids) >= 2:
            self.key_events_frame[
                "first_mt_cut"
            ] = ImpossibleDetection.MORE_THAN_TWO_DAUGHTER_TRACKS
            self.key_events_frame[
                "second_mt_cut"
            ] = ImpossibleDetection.MORE_THAN_TWO_DAUGHTER_TRACKS
            return True

        if not self.mid_body_spots:
            self.key_events_frame["first_mt_cut"] = ImpossibleDetection.NO_MID_BODY_DETECTED
            self.key_events_frame["second_mt_cut"] = ImpossibleDetection.NO_MID_BODY_DETECTED
            return True

        if max(self.mid_body_spots.keys()) < self.key_events_frame["cytokinesis"]:
            self.key_events_frame[
                "first_mt_cut"
            ] = ImpossibleDetection.NO_MID_BODY_DETECTED_AFTER_CYTOKINESIS
            self.key_events_frame[
                "second_mt_cut"
            ] = ImpossibleDetection.NO_MID_BODY_DETECTED_AFTER_CYTOKINESIS
            return True

        return False

    def update_mt_cut_detection(
        self, video, scaler_path, model_path, hmm_bridges_parameters_file
    ) -> None:
        """
        Update micro-tubules cut detection using bridges classification.
        """
        classification_impossible = self.is_bridges_classification_impossible()

        if classification_impossible:
            return

        # Get default classification parameters
        parameters = BridgesClassificationParameters()

        # Perform classification...
        ordered_mb_frames = sorted(self.mid_body_spots.keys())
        first_mb_frame = ordered_mb_frames[0]
        last_mb_frame = ordered_mb_frames[-1]
        first_frame = max(first_mb_frame, self.key_events_frame["cytokinesis"] - 2)  # -2?

        # Load the classifier and scaler
        with open(model_path, "rb") as f:
            classifier: SVC = pickle.load(f)
        # Load the scaler
        with open(scaler_path, "rb") as f:
            scaler: StandardScaler = pickle.load(f)

        list_class_bridges = []
        # Iterate over frames and get the class of the bridge
        for frame in range(first_frame, last_mb_frame + 1):
            min_x = self.position.min_x
            min_y = self.position.min_y

            # Get midbody coordinates
            mb_coords = self.mid_body_spots[frame].position
            x_pos, y_pos = min_x + mb_coords[0], min_y + mb_coords[1]

            # Extract frame image and crop around the midbody Sir-tubulin
            frame_image = video[frame, :, :, :].squeeze().transpose(2, 0, 1)  # C, H, W
            crop = smart_cropping(frame_image, MARGIN, x_pos, y_pos, pad=True)[0, ...]  # H, W

            # Get the class of the bridge
            bridge_class = get_bridge_class(
                crop, scaler, classifier, parameters, plot_enabled=False
            )
            list_class_bridges.append(bridge_class)

        # Make sure cytokinesis bridge is detected as A (no MT cut)
        relative_cytokinesis_frame = self.key_events_frame["cytokinesis"] - first_frame
        if relative_cytokinesis_frame < 0 or list_class_bridges[relative_cytokinesis_frame] != 0:
            self.key_events_frame["first_mt_cut"] = ImpossibleDetection.MT_CUT_AT_CYTOKINESIS
            self.key_events_frame["second_mt_cut"] = ImpossibleDetection.MT_CUT_AT_CYTOKINESIS
            return

        # Read HMM parameters
        if not os.path.exists(hmm_bridges_parameters_file):
            raise FileNotFoundError(f"File {hmm_bridges_parameters_file} not found")
        hmm_parameters = np.load(hmm_bridges_parameters_file)

        # Correct the sequence with HMM
        seq_after_hmm = apply_hmm(hmm_parameters, list_class_bridges)

        # Get index of first element > 0 in sequence (first MT cut)
        first_mt_cut_frame_rel = next((i for i, x in enumerate(seq_after_hmm) if x > 0), -1)

        # Ignore if no MT cut detected
        if first_mt_cut_frame_rel == -1:
            self.key_events_frame["first_mt_cut"] = ImpossibleDetection.NO_CUT_DETECTED
            self.key_events_frame["second_mt_cut"] = ImpossibleDetection.NO_CUT_DETECTED
            return

        first_mt_cut_frame_abs = first_frame + first_mt_cut_frame_rel
        if self.light_spot_detected(video, first_mt_cut_frame_abs):
            self.key_events_frame["first_mt_cut"] = ImpossibleDetection.LIGHT_SPOT
            self.key_events_frame["second_mt_cut"] = ImpossibleDetection.LIGHT_SPOT
            return

        # Update mitosis track accordingly
        self.key_events_frame["first_mt_cut"] = first_mt_cut_frame_abs

        # Get index of first element > 1 in sequence (second MT cut)
        second_mt_cut_frame_rel = next((i for i, x in enumerate(seq_after_hmm) if x > 1), -1)

        # get the frame of the second MT cut
        if second_mt_cut_frame_rel == -1:
            self.key_events_frame["second_mt_cut"] = ImpossibleDetection.NO_CUT_DETECTED
            return

        second_mt_cut_frame_abs = first_frame + second_mt_cut_frame_rel

        # Update mitosis track accordingly
        self.key_events_frame["second_mt_cut"] = second_mt_cut_frame_abs
