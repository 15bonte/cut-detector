from enum import IntEnum


class ImpossibleDetection(IntEnum):
    """
    Enum to represent the different types of impossible detections.
    """

    NORMAL = 0
    NO_MID_BODY_DETECTED = -1
    MORE_THAN_TWO_DAUGHTER_TRACKS = -2
    NEAR_BORDER = -3
    NO_MID_BODY_DETECTED_AFTER_CYTOKINESIS = -4
    MT_CUT_AT_CYTOKINESIS = -5
    NO_CUT_DETECTED = -6
    LIGHT_SPOT = -7
