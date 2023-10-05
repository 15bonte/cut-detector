from .trackmate_spot import TrackMateSpot


class TrackMateFrameSpots:
    """
    Parse TrackMate frame spots from xml file.
    """

    def __init__(self, trackmate_frame_spot, raw_video_shape: list[int]):
        self.spots: list[TrackMateSpot] = []
        for spot in trackmate_frame_spot["Spot"]:
            self.spots.append(TrackMateSpot(spot, raw_video_shape))

        self.frame = int(trackmate_frame_spot["@frame"])
