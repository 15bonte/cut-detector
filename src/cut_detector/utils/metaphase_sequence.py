from __future__ import annotations


class MetaphaseSequence:
    """Class representing a sequence of metaphase frames.

    Parameters
    ----------
    sequence : list[int]
        Sequence of metaphase absolute frames.
    track_id : int
        Track id.
    """

    def __init__(self, sequence: list[int], track_id: int):
        self.first_frame = min(sequence)
        self.last_frame = max(sequence)

        self.track_id = track_id

    def is_mother_candidate(self, frame: int, frames_around_metaphase: int):
        """Check if the frame can be the first one of a daughter frame form this metaphase.
        Returns True if the frame is close to the last frame of the metaphase.

        Parameters
        ----------
        frame : int
            Frame number.
        frames_around_metaphase : int
            Range to look for metaphase candidate spots.

        Returns
        -------
        bool
            Whether the frame is a daughter candidate.
        """
        return (
            abs(self.last_frame - frame) < frames_around_metaphase
            and frame >= self.first_frame
        )

    def is_same(self, metaphase_sequence: MetaphaseSequence):
        """Check if the metaphase sequences are the same.

        Parameters
        ----------
        metaphase_sequence : MetaphaseSequence
            Metaphase sequence.

        Returns
        -------
        bool
            Whether the metaphase sequences are the same.
        """
        return (
            self.track_id == metaphase_sequence.track_id
            and self.last_frame == metaphase_sequence.last_frame
        )

    def is_after(self, metaphase_sequence: MetaphaseSequence):
        """Check if the current metaphase sequence is after the input one.

        Parameters
        ----------
        metaphase_sequence : MetaphaseSequence
            Metaphase sequence.

        Returns
        -------
        bool
        """
        return self.first_frame > metaphase_sequence.first_frame
