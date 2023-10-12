from __future__ import annotations

from copy import deepcopy
import json
from typing import Optional, Tuple, Union
import numpy as np
from skimage.morphology import extrema, opening
import xmltodict
from munch import Munch
from scipy import ndimage


from ..constants.annotations import NAMES_DICTIONARY
from ..constants.tracking import (
    CYTOKINESIS_DURATION,
    FRAMES_AROUND_METAPHASE,
    METAPHASE_INDEX,
    MINIMUM_DISTANCE_TO_BORDER,
)
from .mid_body_spot import MidBodySpot
from .trackmate_track import TrackMateTrack
from .box_dimensions_dln import BoxDimensionsDln
from .box_dimensions import BoxDimensions
from .bridges_classification.impossible_detection import ImpossibleDetection
from .image_tools import resize_image, smart_cropping


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

    def get_mother_daughters_tracks(
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
        mother_track, daughter_tracks = self.get_mother_daughters_tracks(trackmate_tracks)

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
        mother_track, daughter_tracks = self.get_mother_daughters_tracks(trackmate_tracks)

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
        mother_track, daughter_tracks = self.get_mother_daughters_tracks(trackmate_tracks)

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

    def generate_mitosis_summary(self, raw_tracks: list[TrackMateTrack], save_path: str) -> None:
        """
        Unused so far.
        Might be improved with all useful information, saved to csv...
        """
        mitosis_summary = {}

        mother_track, daughter_tracks = self.get_mother_daughters_tracks(raw_tracks)
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
        length_light_spot: int,
        crop_size_light_spot: int,
        h_maxima_light_spot: int,
        intensity_threshold_light_spot: int,
        center_tolerance_light_spot: int,
        min_percentage_light_spot: float,
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
        for i in range(-length_light_spot // 2, length_light_spot // 2):
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
            crop = smart_cropping(img, crop_size_light_spot, x_pos, y_pos, pad=True)[
                0, ...
            ]  # H, W

            # Perform opening to remove small spots and apply h_maxima to get potential spots
            filtered_image = opening(crop, footprint=np.ones((3, 3)))
            local_maxima = extrema.h_maxima(filtered_image, h_maxima_light_spot)

            # Label spot regions and remove inconsistent ones
            labeled_local_maxima, nb_labels = ndimage.label(
                local_maxima, structure=np.ones((3, 3))
            )
            for label in range(1, nb_labels + 1):
                # Labels intensity in original image has to be higher than threshold
                if (
                    crop[np.where(labeled_local_maxima == label)].mean()
                    < intensity_threshold_light_spot
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
                if (np.abs(spot[0] - crop_size_light_spot) < center_tolerance_light_spot) and (
                    np.abs(spot[1] - crop_size_light_spot) < center_tolerance_light_spot
                ):
                    spots = np.delete(spots, np.where((spots == spot).all(axis=1))[0], axis=0)

            if len(spots) > 0:
                nb_spot_detected += 1
            frame_counted += 1

        # Light spot is considered as detected if at least in MIN_PERCENTAGE_LIGHTSPOT % of frames
        if frame_counted > 0:
            percentage_spot_detected = nb_spot_detected / frame_counted
            spot_detected = percentage_spot_detected >= min_percentage_light_spot
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
