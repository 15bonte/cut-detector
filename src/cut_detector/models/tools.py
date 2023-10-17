import os
import urllib.request
from cellpose.utils import download_url_to_file

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_model_path(model_name: str) -> None:
    """
    Returns absolute path to model file.
    """
    if model_name == "segmentation_model":
        path_end = "segmentation_model"
    elif model_name == "metaphase_model":
        path_end = os.path.join("metaphase_cnn", "metaphase_cnn.pt")
    elif model_name == "hmm_metaphase_parameters":
        path_end = "hmm_metaphase_parameters.npz"
    elif model_name == "hmm_bridges_parameters":
        path_end = "hmm_bridges_parameters.npz"
    elif model_name == "svc_scaler":
        path_end = os.path.join("svc_bridges", "scaler.pkl")
    elif model_name == "svc_model":
        path_end = os.path.join("svc_bridges", "model.pkl")
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    local_path = os.path.join(CURRENT_DIR, path_end)

    if not os.path.exists(local_path):
        print(f"Downloading model {model_name}...")
        urllib.request.urlretrieve(
            f"https://raw.githubusercontent.com//15bonte/cut-detector/main/models/{path_end}",
            local_path,
        )

    return local_path
