from .cell_spot import CellSpot


class TrackMateSpot(CellSpot):
    """
    Parse TrackMate spot from xml file.
    """

    def __init__(self, trackmate_spot, raw_video_shape: list[int]):
        x = float(trackmate_spot["@POSITION_X"])
        y = float(trackmate_spot["@POSITION_Y"])
        frame = int(trackmate_spot["@FRAME"])
        id_number = int(trackmate_spot["@ID"])

        # Get min and max positions
        raw_positions: str = trackmate_spot["#text"].split(" ")
        rel_positions_x = [
            float(raw_positions[i]) for i in range(0, len(raw_positions), 2)
        ]
        rel_positions_y = [
            float(raw_positions[i]) for i in range(1, len(raw_positions), 2)
        ]

        rel_min_x, rel_max_x = min(rel_positions_x), max(rel_positions_x)
        rel_min_y, rel_max_y = min(rel_positions_y), max(rel_positions_y)

        # Get spot points
        positions_x = [
            int(x + float(raw_positions[i]))
            for i in range(0, len(raw_positions), 2)
        ]
        positions_y = [
            int(y + float(raw_positions[i]))
            for i in range(1, len(raw_positions), 2)
        ]
        spot_points = [[x, y] for x, y in zip(positions_x, positions_y)]

        super().__init__(
            frame,
            x,
            y,
            id_number,
            rel_min_x,
            rel_max_x,
            rel_min_y,
            rel_max_y,
            spot_points,
        )

        # Clip to video size
        self.abs_min_x, self.abs_max_x = (
            max(self.abs_min_x, 0),
            min(self.abs_max_x, raw_video_shape[2]),
        )
        self.abs_min_y, self.abs_max_y = (
            max(self.abs_min_y, 0),
            min(self.abs_max_y, raw_video_shape[1]),
        )
