import os
import urllib.request

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_model_path(model_name: str) -> None:
    """
    Returns absolute path to model file.
    """
    if model_name == "segmentation":
        model_name = "segmentation_v120"
        files = ["segmentation_model"]
    elif model_name == "metaphase_cnn":
        model_name = "metaphase_cnn_v012"
        files = ["mean_std.json", "metaphase_cnn.pt", "parameters.csv"]
    elif model_name == "hmm":
        files = ["hmm_metaphase_parameters.npz", "hmm_bridges_parameters.npz"]
    elif model_name == "svc_bridges":
        files = ["scaler.pkl", "model.pkl"]
    elif model_name == "bridges_mt_cnn":
        model_name = "bridges_mt_cnn_v009"
        files = ["mean_std.json", "bridges_mt_cnn.pt", "parameters.csv"]
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    local_path = os.path.join(CURRENT_DIR, model_name)
    os.makedirs(local_path, exist_ok=True)

    for file in files:
        file_local_path = os.path.join(local_path, file)
        if not os.path.exists(file_local_path):
            print(f"Downloading data {model_name}...")
            urllib.request.urlretrieve(
                f"https://raw.githubusercontent.com//15bonte/cut-detector-models/main/models/{model_name}/{file}",
                file_local_path,
            )

    return local_path
