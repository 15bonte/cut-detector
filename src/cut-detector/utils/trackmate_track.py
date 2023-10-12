from __future__ import annotations
from typing import Optional, Tuple
from scipy.spatial import ConvexHull, Delaunay
import numpy as np

from ..constants.tracking import (
    FRAMES_AROUND_METAPHASE,
    INTERPHASE_INDEX,
    METAPHASE_INDEX,
    MAX_FRAME_GAP,
)
from .box_dimensions import BoxDimensions
from .box_dimensions_dln import BoxDimensionsDln
from .trackmate_spot import TrackMateSpot
from .trackmate_frame_spots import TrackMateFrameSpots


def get_whole_box_dimensions_dln(
    tracks: list[TrackMateTrack], frame: int
) -> Tuple[BoxDimensionsDln, list[list[int]]]:
    box_dimensions_dln = BoxDimensionsDln()
    track_frame_points = []

    # For all tracks: mother and daughter(s)
    for track in tracks:
        if frame in track.track_spots:
            current_spot = track.track_spots[frame]
            track_frame_points = track_frame_points + current_spot.spot_points
            box_dimensions_dln.update(
                current_spot.abs_min_x,
                current_spot.abs_max_x,
                current_spot.abs_min_y,
                current_spot.abs_max_y,
            )
    return box_dimensions_dln, track_frame_points


class TrackMateTrack:
    """
    Parse TrackMate track from xml file.
    """

    def __init__(self, trackmate_track):
        self.track_start = int(float(trackmate_track["@TRACK_START"]))
        self.track_id = int(trackmate_track["@TRACK_ID"])

        self.track_spots_ids: set[int] = set()

        if isinstance(trackmate_track["Edge"], dict):  # only one edge
            self.track_spots_ids.add(int(trackmate_track["Edge"]["@SPOT_SOURCE_ID"]))
            self.track_spots_ids.add(int(trackmate_track["Edge"]["@SPOT_TARGET_ID"]))
        else:
            for edge in trackmate_track["Edge"]:
                self.track_spots_ids.add(int(edge["@SPOT_SOURCE_ID"]))
                self.track_spots_ids.add(int(edge["@SPOT_TARGET_ID"]))

        self.start = int(float(trackmate_track["@TRACK_START"]))
        self.stop = int(float(trackmate_track["@TRACK_STOP"]))

        self.track_spots: dict[int, TrackMateSpot] = {}  # {frame: TrackMateSpot}
        self.metaphase_spots: list[TrackMateSpot] = []

        # Can be different from len(self.track_spots) if we have a gap in the track
        self.number_spots = 0

    def update_metaphase_spots(self, predictions: list[int]) -> None:
        """
        Populate metaphase_spots with candidates.

        Parameters
        ----------
        predictions : list[int]
            list of predictions for each frame of the track.

        """
        for idx, frame in enumerate(sorted(self.track_spots.keys())):
            self.track_spots[frame].predicted_phase = predictions[idx]

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
                frame in self.track_spots  # current frame contains a spot
                and predictions[frame - self.start]
                == METAPHASE_INDEX  # current spot is in metaphase
                and frame != self.stop  # current spot is not last spot
                and (
                    predictions[frame - self.start + 1] != METAPHASE_INDEX
                    and frame in self.track_spots
                )  # next frame is not a spot in metaphase
            ):
                self.metaphase_spots.append(self.track_spots[frame])

    def has_close_metaphase(self, spot: TrackMateSpot, target_frame: int) -> bool:
        """
        Returns True if corresponding track contains one metaphase spot close to target frame.
        """
        # Look for metaphase spot
        for metaphase_spot in self.metaphase_spots:
            if abs(metaphase_spot.frame - target_frame) < FRAMES_AROUND_METAPHASE:
                # Mother track found!
                spot.corresponding_metaphase_spot = metaphase_spot
                return True
        return False

    def compute_metaphase_iou(self, daughter_track: TrackMateTrack) -> float:
        """
        Get intersection between daughter area at first frame and self area at previous frame.
        Get self area at previous frame.
        Returns the quotient.

        Ideally, it should be close to 0.5 as a daughter cell should occupy 50% of the area
        of the mother cell.

        May be improved by checking overlap of areas instead of convex hulls.
        """

        daughter_track_first_frame = min(daughter_track.track_spots.keys())

        # Compute two regions
        self_previous_region = self.compute_dln_from_tracks(
            daughter_track_first_frame - 1, relative=False
        )
        daughter_region = daughter_track.compute_dln_from_tracks(
            daughter_track_first_frame, relative=False
        )

        local_shape = (
            max(self_previous_region.max_y, daughter_region.max_y),
            max(self_previous_region.max_x, daughter_region.max_x),
        )

        indices = np.stack(
            np.indices(local_shape),
            axis=-1,
        )

        self_previous_out_idx = np.nonzero(self_previous_region.dln.find_simplex(indices) + 1)
        self_previous_region_mask = np.zeros(local_shape, dtype=bool)
        self_previous_region_mask[self_previous_out_idx] = True

        daughter_out_idx = np.nonzero(daughter_region.dln.find_simplex(indices) + 1)
        daughter_region_mask = np.zeros(local_shape, dtype=bool)
        daughter_region_mask[daughter_out_idx] = True

        # Compute intersection of both regions
        overlap = np.sum(self_previous_region_mask * daughter_region_mask)

        # Compute previous region area
        previous_area = np.sum(self_previous_region_mask)

        return overlap / previous_area

    def compute_dln_from_tracks(
        self,
        frame: int,
        previous_box_dimensions_dln: Optional[BoxDimensionsDln] = None,
        additional_tracks: Optional[list[TrackMateTrack]] = None,
        relative: bool = True,
    ) -> BoxDimensionsDln:
        tracks = [self]
        if additional_tracks is not None:
            tracks = tracks + additional_tracks

        box_dimensions_dln, track_frame_points = get_whole_box_dimensions_dln(tracks, frame)

        # If missing spot at this frame...
        if box_dimensions_dln.is_empty():
            # ... use previous frame data if provided
            if previous_box_dimensions_dln:
                return previous_box_dimensions_dln

            # ... or try with previous frame if not
            for _ in range(MAX_FRAME_GAP):
                frame = frame - 1
                (
                    box_dimensions_dln,
                    track_frame_points,
                ) = get_whole_box_dimensions_dln(tracks, frame)
                if not box_dimensions_dln.is_empty():
                    break

        # Should not be empty after this loop
        if box_dimensions_dln.is_empty():
            raise ValueError(
                f"No previous dln & Tracks with no spots in {MAX_FRAME_GAP} frames in a row"
            )

        # Else, compute convex hull and Delaunay triangulation
        # Switch dimensions
        if relative:
            track_frame_points = [
                [y - box_dimensions_dln.min_y, x - box_dimensions_dln.min_x]
                for x, y in track_frame_points
            ]
        else:
            track_frame_points = [[y, x] for x, y in track_frame_points]
        # Compute hull
        hull = ConvexHull(points=track_frame_points)
        box_dimensions_dln.dln = Delaunay(np.array(track_frame_points)[hull.vertices])

        return box_dimensions_dln

    def get_spots_data(
        self, raw_spots: list[TrackMateFrameSpots], raw_video: np.array
    ) -> list[np.array]:
        """
        Parse xml data to assign spots at corresponding track.
        Returns list of nucleus crops for each frame.

        Parameters
        ----------
        track: TrackMateTrack
        raw_spots: [TrackMateFrameSpots]
        raw_video: T, H, W, C

        Returns
        -------
        nucleus_crops: [C, H, W]
        """

        spot_abs_positions = {}  # {frame: BoxDimensions}
        nucleus_crops = []  # [C, H, W]

        for frame_spots in raw_spots:
            # Ignore spots before or after current track
            if frame_spots.frame < self.start or frame_spots.frame > self.stop:
                continue
            for spot in frame_spots.spots:
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
                self.track_spots[spot.frame] = spot

        # If no spot in track for current frame, use previous frame position
        for frame in range(self.start, self.stop + 1):
            if frame in spot_abs_positions:
                min_y, max_y, min_x, max_x = (
                    spot_abs_positions[frame].min_y,
                    spot_abs_positions[frame].max_y,
                    spot_abs_positions[frame].min_x,
                    spot_abs_positions[frame].max_x,
                )
            nucleus = raw_video[frame, min_y:max_y, min_x:max_x, :]  # H, W, C
            nucleus = np.moveaxis(nucleus, -1, 0)  # C, H, W
            nucleus_crops.append(nucleus)

        self.number_spots = len(nucleus_crops)
        return nucleus_crops
