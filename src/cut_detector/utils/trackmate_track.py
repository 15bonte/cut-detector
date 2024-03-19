from .cell_track import CellTrack


class TrackMateTrack(CellTrack):
    """
    Parse TrackMate track from xml file.
    """

    def __init__(self, trackmate_track):
        track_id = int(trackmate_track["@TRACK_ID"])

        track_spots_ids: set[int] = set()

        if isinstance(trackmate_track["Edge"], dict):  # only one edge
            track_spots_ids.add(
                int(trackmate_track["Edge"]["@SPOT_SOURCE_ID"])
            )
            track_spots_ids.add(
                int(trackmate_track["Edge"]["@SPOT_TARGET_ID"])
            )
        else:
            for edge in trackmate_track["Edge"]:
                track_spots_ids.add(int(edge["@SPOT_SOURCE_ID"]))
                track_spots_ids.add(int(edge["@SPOT_TARGET_ID"]))

        start = int(float(trackmate_track["@TRACK_START"]))
        stop = int(float(trackmate_track["@TRACK_STOP"]))

        super().__init__(track_id, track_spots_ids, start, stop)

    def adapt_deprecated_attributes(self) -> None:
        """
        Adapt deprecated attributes.

        Returns
        -------
        None.

        """
        if hasattr(self, "track_spots"):
            self.spots = self.track_spots
            del self.track_spots
