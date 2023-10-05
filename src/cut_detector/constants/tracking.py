# Tracking parameters (ratio of average spot size)
LINKING_MAX_DISTANCE_RATIO = 1
GAP_CLOSING_MAX_DISTANCE_RATIO = 1 / 2
MAX_FRAME_GAP = 3

# Metaphase links
FRAMES_AROUND_METAPHASE = 10  # look for metaphase candidate spots in this range
MAX_SPOT_DISTANCE_FOR_SPLIT = 20  # look for metaphase candidate spots in that distance
MIN_TRACK_SPOTS = 10  # minimum spots in track to consider it
MINIMUM_METAPHASE_INTERVAL = 10  # Minimum distance between two metaphases

INTERPHASE_INDEX = 0  # interphase index for CNN classification model
METAPHASE_INDEX = 1  # metaphase index for CNN classification model

# Mid-body tracking
MINIMUM_MID_BODY_TRACK_LENGTH = 10  # minimum spots in mid-body track to consider it
WEIGHT_MKLP_INTENSITY_FACTOR = 10  # weight of intensity in spot dist calculation (cf TrackMate)
WEIGHT_SIR_INTENSITY_FACTOR = 3.33  # weight of sir intensity in spot distance calculation
CYTOKINESIS_DURATION = 20  # number of frames to look for mid-body in between cells
MID_BODY_LINKING_MAX_DISTANCE = 100  # maximum distance between two mid-bodies to link them

# Detection using bigfish
SIGMA = 2
THRESHOLD = 1

# Detection using h-maxima
H_MAXIMA_THRESHOLD = 5

# Minimum distance to border to consider is_near_border = true
MINIMUM_DISTANCE_TO_BORDER = 20
