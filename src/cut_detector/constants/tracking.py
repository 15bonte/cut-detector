# Cellpose parameters
FLOW_THRESHOLD = 0.0
CELLPROB_THRESHOLD = 0.0
AUGMENT = True

# Tracking parameters (ratio of average spot size)
LINKING_MAX_DISTANCE_RATIO = 1
GAP_CLOSING_MAX_DISTANCE_RATIO = 1 / 2
MAX_FRAME_GAP = 3

# Metaphase links
FRAMES_AROUND_METAPHASE = 10  # look for metaphase candidate spots in this range

INTERPHASE_INDEX = 0  # interphase index for CNN classification model
METAPHASE_INDEX = 1  # metaphase index for CNN classification model

# Mid-body tracking
CYTOKINESIS_DURATION = 20  # number of frames to look for mid-body in between cells

# Minimum distance to border to consider is_near_border = true
MINIMUM_DISTANCE_TO_BORDER = 20
