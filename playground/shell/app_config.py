import os

from cut_detector.data.tools import get_data_path


EVAL_DATA_DIRS = {
    "Default": os.path.dirname(get_data_path("mitoses")),
    "Standard": "eval_data/Data Standard",
    "Spastin": "eval_data/Data spastin",
    "Cep55": "eval_data/Data cep55",
}

GT_SPOT_SIZE = 6
TEST_SPOT_SIZE = 8
DYN_SPOT_SIZE = 10

MAX_DYN_SPOT_DISPLAYED = 50
