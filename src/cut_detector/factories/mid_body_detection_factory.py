import concurrent.futures
from typing import Optional
import numpy as np
from skimage.morphology import extrema, opening
from scipy import ndimage
from shapely.ops import nearest_points
from shapely import Polygon, Point
from tqdm import tqdm

from ..utils.parameters import Parameters
from ..utils.mid_body_detection.detection import (
    DETECTION_FUNCTIONS,
)
from ..utils.cell_track import CellTrack
from ..utils.mid_body_track import MidBodyTrack
from ..utils.image_tools import smart_cropping
from ..utils.mid_body_spot import MidBodySpot
from ..utils.mitosis_track import MitosisTrack
from ..utils.track_generation import generate_tracks_from_spots
from ..utils.cell_spot import CellSpot
from ..utils.mid_body_detection.tracking import get_tracking_method


class MidBodyDetectionFactory:
    """
    Class to perform mid-body detection, tracking and filtering.

    Parameters
    ----------
    params : Parameters
        Video parameters.
    minimum_mid_body_track_length : int
        Minimum spots in mid-body track to consider it.
    """

    def __init__(
        self,
        params: Parameters,
        minimum_mid_body_track_length=5,
    ):
        self.params = params
        self.minimum_mid_body_track_length = minimum_mid_body_track_length

    def update_mid_body_spots(
        self,
        mitosis_track: MitosisTrack,
        mitosis_movie: np.ndarray,
        tracks: list[CellTrack],
        parallel_detection: bool,
        detection_method: str,
        tracking_method: str = "spatial_laptrack",
        log_blob_spot: bool = False,
    ) -> None:
        """
        Get spots of best mitosis track.

        Parameters
        ----------
        mitosis_track: MitosisTrack
            Mitosis track to update.
        mitosis_movie: np.ndarray
            Mitosis movie. TYXC.
        tracks: list[CellTrack]
            List of cell tracks.
        parallel_detection: bool
            If True, use parallelization for mid-body spots detection.
        detection_method: str
            Method to detect mid-body spots.
        tracking_method: str
            Method to track mid-body spots.
        log_blob_spot: bool
            If True, display log of spots detected.
        """

        spots_candidates = self.detect_mid_body_spots(
            mitosis_movie,
            method=detection_method,
            parallelization=parallel_detection,
            log_blob_spot=log_blob_spot,
            mitosis_track=mitosis_track,
        )

        mid_body_tracks: list[MidBodyTrack] = generate_tracks_from_spots(
            spots_candidates,
            get_tracking_method(
                tracking_method, self.params.spatial_resolution
            ),
        )

        kept_track = self._select_best_track(
            mitosis_track,
            mid_body_tracks,
            tracks,
            mitosis_movie[..., self.params.sir_channel],
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
        method: str,
        parallelization: bool = False,
        log_blob_spot: bool = False,
        mitosis_track: Optional[MitosisTrack] = None,
    ) -> dict[int, list[MidBodySpot]]:
        """
        Detect mid-body spots in mitosis movie.

        Parameters
        ----------
        mitosis_movie: np.ndarray
            Mitosis movie. TYXC.
        method: str
            Method to detect mid-body spots.
        parallelization: bool
            If True, use parallelization for mid-body spots detection.
        log_blob_spot: bool
            If True, display log of spots detected.
        mitosis_track: MitosisTrack
            Mitosis track to get mask positions.

        Returns
        ----------
        dict[int, list[MidBodySpot]]
            Dictionary of mid-body spots per frame.
        """
        assert isinstance(parallelization, bool)

        if parallelization:
            mid_bodies = self.thread_pool_detect_mid_body_spots(
                mitosis_movie,
                method,
                mitosis_track,
            )

        else:
            mid_bodies = self.serial_detect_mid_body_spots(
                mitosis_movie,
                method,
                log_blob_spot,
                mitosis_track,
            )

        # Return a sorted dictionary to ensure tracking consistency
        ordered_mid_bodies = dict(sorted(mid_bodies.items()))

        return ordered_mid_bodies

    def serial_detect_mid_body_spots(
        self,
        mitosis_movie: np.ndarray,
        method: str,
        log_blob_spot: bool = False,
        mitosis_track: Optional[MitosisTrack] = None,
    ) -> dict[int, list[MidBodySpot]]:
        """Detect mid-body spots in mitosis movie.

        Parameters
        ----------
        mitosis_movie: np.ndarray
            Mitosis movie. TYXC.
        mode: str
            Method to detect mid-body spots.
        log_blob_spot: bool
            If True, display log of spots detected.
        mitosis_track: MitosisTrack
            Mitosis track to get mask positions.
        """

        spots_dictionary = {}
        nb_frames = mitosis_movie.shape[0]

        print("Detecting mid-body spots...")
        for frame in tqdm(range(nb_frames)):

            mitosis_frame = mitosis_movie[frame]  # YXC
            spots = self._spot_detection(
                mitosis_frame,
                method,
                frame,
                log_blob_spot,
                mitosis_track,
            )

            # Update dictionary
            spots_dictionary[frame] = spots

        return spots_dictionary

    def thread_pool_detect_mid_body_spots(
        self,
        mitosis_movie: np.ndarray,
        method: str,
        mitosis_track: Optional[MitosisTrack] = None,
    ) -> dict[int, list[MidBodySpot]]:
        """Detect mid-body spots in mitosis movie.
        Parallelized version.

        Parameters
        ----------
        mitosis_movie: np.ndarray
            Mitosis movie. TYXC.
        mode: str
            Method to detect mid-body spots.
        log_blob_spot: bool
            If True, display log of spots detected.
        mitosis_track: MitosisTrack
            Mitosis track to get mask positions.
        """

        def frame_spot_detection(frame: int) -> tuple[int, list[MidBodySpot]]:
            return frame, self._spot_detection(
                mitosis_movie[frame],
                method,
                frame,
                mitosis_track=mitosis_track,
            )

        future_list = []
        with concurrent.futures.ThreadPoolExecutor() as e:
            for frame in range(mitosis_movie.shape[0]):
                future_list.append(
                    e.submit(
                        frame_spot_detection,
                        frame,
                    )
                )

        return {
            res.result()[0]: res.result()[1]
            for res in concurrent.futures.as_completed(future_list)
        }

    def _spot_detection(
        self,
        image: np.ndarray,
        method: str,
        frame: int,
        log_blob_spot: bool = False,
        mitosis_track: Optional[MitosisTrack] = None,
    ) -> list[MidBodySpot]:
        """Perform spot detection on a single frame of a movie.

        Parameters
        ----------
        image: np.ndarray
            Image to process. YXC.
        method: str
            Method to detect mid-body spots.
        frame: int
            Frame number.
        log_blob_spot: bool
            If True, display log of spots detected.
        mitosis_track: Optional[MitosisTrack]
            Mitosis track to get mask positions.

        Returns
        ----------
        list[MidBodySpot]
            List of detected mid-body spots.
        """

        # Try to load mitosis_track to get mask positions
        if mitosis_track is None:
            image_sir = image[:, :, self.params.sir_channel]
            image_mklp = image[:, :, self.params.mid_body_channel]
            shift_x, shift_y = 0, 0
        elif isinstance(mitosis_track, MitosisTrack):
            mitosis_position = (
                mitosis_track.position
            )  # mitosis position in movie
            frame_position = mitosis_track.contour_positions[
                frame + mitosis_track.min_frame
            ]  # frame position in mitosis
            shift_x = frame_position.min_x - mitosis_position.min_x
            shift_y = frame_position.min_y - mitosis_position.min_y
            image_sir = image[
                shift_y : frame_position.max_y - mitosis_position.min_y,
                shift_x : frame_position.max_x - mitosis_position.min_x,
                self.params.sir_channel,
            ]
            image_mklp = image[
                shift_y : frame_position.max_y - mitosis_position.min_y,
                shift_x : frame_position.max_x - mitosis_position.min_x,
                self.params.mid_body_channel,
            ]
        else:
            raise RuntimeError(
                f"Invalid type for argument mitosis_track: {mitosis_track}"
            )

        if method in DETECTION_FUNCTIONS:
            # Function called referenced by name
            spots = [
                (int(spot[0]), int(spot[1]), int(spot[2]))
                for spot in DETECTION_FUNCTIONS[method](image_mklp)
            ]

            if log_blob_spot:
                for s in spots:
                    print(f"found x:{s[1]}  y:{s[0]}  s:{s[2]}")

        elif method == "h_maxima":  # NB: old method, to be removed
            h_maxima_threshold = 5.0
            # Perform opening followed by closing to remove small spots
            filtered_image = opening(image_mklp, footprint=np.ones((3, 3)))
            # Get local maxima using h_maxima
            local_maxima = extrema.h_maxima(filtered_image, h_maxima_threshold)
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
            raise ValueError(f"Unknown mode: [{method}]")

        # Spots can be a list of tuples with 2 or 3 values: 2 values: (y, x) if
        # h_maxima, 3 values: (y, x, sigma) if any blob-based method used
        mid_body_spots = [
            MidBodySpot(
                frame,
                x=position[1] + shift_x,  # switch (y, x) to (x, y)
                y=position[0] + shift_y,
                intensity=self._get_average_intensity(position, image_mklp),
                sir_intensity=self._get_average_intensity(position, image_sir),
            )
            for position in spots
        ]

        return mid_body_spots

    @staticmethod
    def _get_average_intensity(
        position: tuple[int], image: np.ndarray, margin=1
    ) -> int:
        """Get average intensity of a spot in an image.

        Parameters
        ----------
        position : tuple[int]
            Spot position. (y, x).
        image : np.ndarray
            Image to process. YX.
        margin : int
            Margin around the spot to consider.

        Returns
        ----------
        int
            Average intensity of the spot.
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

    def _get_mid_body_expected_positions(
        self,
        mitosis_track: MitosisTrack,
        cell_tracks: list[CellTrack],
    ) -> tuple[dict, dict[int, CellSpot], dict[int, CellSpot]]:
        """Compute Mid-body expected positions for first cytokinesis frames.
        Defined at the point where the two cells are the closest.
        Outputs are relative to the mitosis_track position.

        Parameters
        ----------
        mitosis_track: MitosisTrack
            Mitosis track to get mother and daughter tracks.
        cell_tracks: list[CellTrack]
            List of cell tracks.

        Returns
        ----------
        tuple[dict, dict[int, CellSpot], dict[int, CellSpot]]
            Expected positions, mother spots, daughter spots.
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
            daughter_track.start + self.params.cytokinesis_duration,
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
        cell_tracks: list[CellTrack],
        tubulin_movie: np.ndarray,
    ) -> MidBodyTrack:
        """Select best track from mid-body tracks.

        Parameters
        ----------
        mitosis_track: MitosisTrack
            Mitosis track to get mother and daughter tracks.
        mid_body_tracks: list[MidBodyTrack]
            List of mid-body tracks.
        cell_tracks: list[CellTrack]
            List of cell tracks.
        tubulin_movie: np.ndarray
            Tubulin movie. TYX.

        Returns
        -------
        MidBodyTrack
            Best mid-body track.
        """

        # Filter: keep only tracks with sir-tubulin signal at cytokinesis
        # Define range of frames to study
        abs_min_frame = mitosis_track.key_events_frame[
            "no_mt_cut"
        ]  # Cytokinesis start
        abs_max_frame = abs_min_frame + int(
            self.params.cytokinesis_duration / 2
        )
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
        expected_positions, _, _ = self._get_mid_body_expected_positions(
            mitosis_track, cell_tracks
        )
        # Get list of expected distances
        expected_distances = []
        for track in kept_tracks:
            expected_distances.append(
                track.get_expected_distance(
                    expected_positions, self.params.spatial_resolution
                )
            )
        assert len(expected_distances) == len(kept_tracks)

        # Sort tracks by expected distance
        sorted_tracks = sorted(
            kept_tracks,
            key=lambda track: expected_distances[kept_tracks.index(track)],
        )
        return sorted_tracks[0]
