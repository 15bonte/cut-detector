import os
import urllib.request

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_model_path(model_name: str) -> None:
    """
    Returns absolute path to model file.
    """
    sub_folder_to_create = None
    if model_name == "segmentation_model":
        path_end = "segmentation_model"
    elif model_name == "metaphase_model":
        path_end = "metaphase_cnn/metaphase_cnn.pt"
        sub_folder_to_create = "metaphase_cnn"
    elif model_name == "hmm_metaphase_parameters":
        path_end = "hmm_metaphase_parameters.npz"
    elif model_name == "hmm_bridges_parameters":
        path_end = "hmm_bridges_parameters.npz"
    elif model_name == "svc_scaler":
        path_end = "svc_bridges/scaler.pkl"
        sub_folder_to_create = "svc_bridges"
    elif model_name == "svc_model":
        path_end = "svc_bridges/model.pkl"
        sub_folder_to_create = "svc_bridges"
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    local_path = os.path.join(CURRENT_DIR, path_end)

    if not os.path.exists(local_path):
        print(f"Downloading model {model_name}...")
        if sub_folder_to_create is not None:
            os.makedirs(os.path.join(CURRENT_DIR, sub_folder_to_create), exist_ok=True)
        urllib.request.urlretrieve(
            f"https://raw.githubusercontent.com//15bonte/cut-detector-models/main/models/{path_end}",
            local_path,
        )

    return local_path
