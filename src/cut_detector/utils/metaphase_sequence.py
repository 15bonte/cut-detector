class MetaphaseSequence:
    """Class representing a sequence of metaphase frames.

    Parameters
    ----------
    sequence : list[int]
        Sequence of metaphase frames.
    """

    def __init__(self, sequence: list[int]):
        self.first_frame = min(sequence)
        self.last_frame = max(sequence)
