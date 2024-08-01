from __future__ import annotations
from typing import Optional
from tqdm import tqdm
import numpy as np
import pandas as pd

from ..constants.tracking import (
    FRAMES_AROUND_METAPHASE,
    INTERPHASE_INDEX,
    METAPHASE_INDEX,
)
from .track import Track
from .box_dimensions_contour import BoxDimensionsContour
from .box_dimensions import BoxDimensions
from .cell_spot import CellSpot


def get_whole_box_dimensions_advanced(
    tracks: list[CellTrack], frame: int
) -> BoxDimensionsContour:
    """
    Merge different tracks.

    Parameters
    ----------
    predictions : list[int]
        list of predictions for each frame of the track.

    Returns
    -------
    BoxDimensionsContour : Box dimension of merged tracks.

    """
    box_dimensions_contour = BoxDimensionsContour()

    # For all tracks: mother and daughter(s)
    for track in tracks:
        if frame in track.spots:
            current_spot = track.spots[frame]
            box_dimensions_contour.list_points.append(current_spot.spot_points)
            box_dimensions_contour.update(
                current_spot.abs_min_x,
                current_spot.abs_max_x,
                current_spot.abs_min_y,
                current_spot.abs_max_y,
            )
    return box_dimensions_contour


def generate_tracking_movie(
    tracks: list[CellTrack], video: np.ndarray
) -> np.ndarray:
    """
    Generate tracking movie.

    Parameters
    ----------
    tracks : list[CellTrack]
        List of tracks.
    video : np.ndarray
        Video. TYXC.

    Returns
    -------
    np.ndarray
        Tracking movie. TYX.
    """
    nb_frames, height, width = video.shape[:-1]

    mask = np.zeros((nb_frames, width, height)).astype(np.uint8)  # TXY
    for frame in tqdm(range(nb_frames)):
        for track in tracks:
            box_dim_contours = get_whole_box_dimensions_advanced(
                [track], frame
            )  # (x, y)
            mask[frame] = np.maximum(
                mask[frame],
                int(track.track_id + 1)
                * box_dim_contours.get_mask((width, height)),
            )

    # Switch X and Y axes to match video shape
    mask = np.moveaxis(mask, -1, 1)  # TYX
    return mask


class CellTrack(Track[CellSpot]):
    """
    Cell track.
    """

    max_frame_gap = 3

    def __init__(
        self, track_id: int, track_spots_ids: set[int], start: int, stop: int
    ) -> None:
        super().__init__(track_id)
        self.track_spots_ids = track_spots_ids
        self.start = start
        self.stop = stop

        self.metaphase_spots: list[CellSpot] = []

    @classmethod
    def from_spots(cls, track_id: int, spots: list[CellSpot]) -> CellTrack:
        """
        Create a CellTrack from a list of CellSpot.

        Parameters
        ----------
        track_id : int
            Track identifier.
        """
        track_spots_ids = set([spot.id for spot in spots])
        start = min([spot.frame for spot in spots])
        stop = max([spot.frame for spot in spots])
        track = cls(track_id, track_spots_ids, start, stop)
        for spot in spots:
            track.add_spot(spot)
        return track

    def update_metaphase_spots(self, predictions: list[int]) -> None:
        """
        Populate metaphase_spots with candidates.

        Parameters
        ----------
        predictions : list[int]
            list of predictions for each frame of the track.

        Returns
        -------
        None.

        """
        for idx, frame in enumerate(sorted(self.spots.keys())):
            self.spots[frame].predicted_phase = predictions[idx]

        # Store last metaphase spot of each group
        metaphase_finished = False
        for frame in range(self.start, self.stop + 1):
            # Ignore first spots of cell as they are metaphase only if end of previous metaphase
            if predictions[frame - self.start] == INTERPHASE_INDEX:
                metaphase_finished = True
            if not metaphase_finished:
                continue

            # From this point, get metaphase spots
            if (
                frame in self.spots  # current frame contains a spot
                and predictions[frame - self.start]
                == METAPHASE_INDEX  # current spot is in metaphase
                and frame != self.stop  # current spot is not last spot
                and (
                    predictions[frame - self.start + 1] != METAPHASE_INDEX
                    and frame in self.spots
                )  # next frame is not a spot in metaphase
            ):
                self.metaphase_spots.append(self.spots[frame])

    def has_close_metaphase(self, spot: CellSpot, target_frame: int) -> bool:
        """
        Parameters
        ----------
        spot : CellSpot
            Only used to update corresponding metaphase spot.

        Returns
        -------
        bool : True if corresponding track contains one metaphase spot close to target frame.

        """
        # Look for metaphase spot
        for metaphase_spot in self.metaphase_spots:
            if (
                abs(metaphase_spot.frame - target_frame)
                < FRAMES_AROUND_METAPHASE
            ):
                # Mother track found!
                spot.corresponding_metaphase_spot = metaphase_spot
                return True
        return False

    def compute_metaphase_iou(self, daughter_track: CellTrack) -> float:
        """
        Get intersection between daughter area at first frame and self area at previous frame.
        Get self area at previous frame. Returns the quotient.

        Ideally, it should be close to 0.5 as a daughter cell should occupy 50% of the area
        of the mother cell.

        Parameters
        ----------
        daughter_track : CellTrack
            Potential daughter track.

        Returns
        -------
        float
            Intersection Over Union.
        """

        daughter_track_first_frame = min(daughter_track.spots.keys())

        # If current track starts at first frame, ignore as it cannot be a mother track
        if self.start == daughter_track_first_frame:
            return -1

        # Compute two regions
        self_previous_region = self.compute_contour_from_tracks(
            daughter_track_first_frame - 1, relative=False
        )
        daughter_region = daughter_track.compute_contour_from_tracks(
            daughter_track_first_frame, relative=False
        )

        local_shape = (
            max(self_previous_region.max_y, daughter_region.max_y),
            max(self_previous_region.max_x, daughter_region.max_x),
        )

        # Compute masks
        self_previous_region_mask = self_previous_region.get_mask(local_shape)
        daughter_region_mask = daughter_region.get_mask(local_shape)

        # Compute intersection of both regions
        overlap = np.sum(self_previous_region_mask * daughter_region_mask)

        # Compute previous region area
        previous_area = np.sum(self_previous_region_mask)

        return overlap / previous_area

    def compute_contour_from_tracks(
        self,
        frame: int,
        previous_box_dimensions_contour: Optional[BoxDimensionsContour] = None,
        additional_tracks: Optional[list[CellTrack]] = None,
        relative: bool = True,
    ) -> BoxDimensionsContour:
        """
        Compute contours at given frame.

        Parameters
        ----------
        frame : int
            Frame at which contour is computed.
        previous_box_dimensions_contour : Optional[BoxDimensionsContour] = None
            If specified and current track has no spot at frame, no computation is done
            and previous_box_dimensions_contour is returned.
        additional_tracks : Optional[list[CellTrack]] = None
            If specified, perform computation on both current track and these additional tracks.
        relative : bool = True
            If True, indexes are relative to current track position.

        Returns
        -------
        BoxDimensionsContour : Track(s) contour.

        """
        tracks = [self]
        if additional_tracks is not None:
            tracks = tracks + additional_tracks

        box_dimensions_advanced = get_whole_box_dimensions_advanced(
            tracks, frame
        )

        # If missing spot at this frame...
        if box_dimensions_advanced.is_empty():
            # ... use previous frame data if provided
            if previous_box_dimensions_contour:
                return previous_box_dimensions_contour

            # ... or try with previous frame if not
            for _ in range(CellTrack.max_frame_gap):
                frame = frame - 1
                box_dimensions_advanced = get_whole_box_dimensions_advanced(
                    tracks, frame
                )
                if not box_dimensions_advanced.is_empty():
                    break

        # Should not be empty after this loop
        if box_dimensions_advanced.is_empty():
            raise ValueError(
                f"No previous box & Tracks with no spots in {CellTrack.max_frame_gap} frames in a row"
            )

        box_dimensions_advanced.update_list_points(relative)

        return box_dimensions_advanced

    def get_spots_data(
        self, raw_spots: list[CellSpot], raw_video: np.ndarray
    ) -> list[np.array]:
        """
        Generate crops around cells for metaphase CNN inference.

        Parameters
        ----------
        raw_spots : list[CellSpot]
            All video spots.
        raw_video : np.ndarray
            TYXC

        Returns
        -------
        list[np.array]: Cell crops (CYX)

        """

        spot_abs_positions = {}  # {frame: BoxDimensions}
        cell_crops = []  # CYX

        for spot in raw_spots:
            # Ignore spots before or after current track
            if spot.frame < self.start or spot.frame > self.stop:
                continue
            # Ignore spots not in current track
            if spot.id not in self.track_spots_ids:
                continue
            # Store positions
            spot_abs_positions[spot.frame] = BoxDimensions(
                spot.abs_min_x,
                spot.abs_max_x,
                spot.abs_min_y,
                spot.abs_max_y,
            )
            # Store all spots
            spot.track_id = self.track_id  # add track information
            self.spots[spot.frame] = spot

        # If no spot in track for current frame, use previous frame position
        for frame in range(self.start, self.stop + 1):
            if frame in spot_abs_positions:
                min_y, max_y, min_x, max_x = (
                    spot_abs_positions[frame].min_y,
                    spot_abs_positions[frame].max_y,
                    spot_abs_positions[frame].min_x,
                    spot_abs_positions[frame].max_x,
                )
            nucleus = raw_video[frame, min_y:max_y, min_x:max_x, :]  # YXC
            nucleus = np.moveaxis(nucleus, -1, 0)  # CYX
            cell_crops.append(nucleus)

        return cell_crops

    @staticmethod
    def track_df_to_track_list(
        track_df: pd.DataFrame,
        spots: dict[int, list[CellSpot]],
    ) -> list[CellTrack]:

        track_df.reset_index(inplace=True)
        track_df.dropna(inplace=True)
        id_to_track = {}

        for _, row in track_df.iterrows():
            track_id = row["track_id"]
            track: list = id_to_track.get(track_id)
            if track is None:
                id_to_track[track_id] = []
                track = id_to_track[track_id]
            frame = row["frame"]
            idx_in_frame = row["idx_in_frame"]
            track.append(spots[int(frame)][int(idx_in_frame)])
        return [
            CellTrack.from_spots(track_id, spots)
            for track_id, spots in enumerate(id_to_track.values())
        ]
