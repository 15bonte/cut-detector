# Metaphase links
FRAMES_AROUND_METAPHASE = (
    10  # look for metaphase candidate spots in this range
)

INTERPHASE_INDEX = 0  # interphase index for CNN classification model
METAPHASE_INDEX = 1  # metaphase index for CNN classification model

# Mid-body tracking
CYTOKINESIS_DURATION = (
    20  # number of frames to look for mid-body in between cells
)

# Time resolution
TIME_RESOLUTION = 10  # 1 frame = 10 minutes

# Channels
MID_BODY_CHANNEL = 1
SIR_CHANNEL = 0
