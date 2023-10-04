import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_model_path(model_name: str) -> None:
    """
    Returns absolute path to model file.
    """
    if model_name == "metaphase_model":
        return os.path.join(CURRENT_DIR, "metaphase_cnn", "metaphase_cnn.pt")
    if model_name == "hmm_metaphase_parameters":
        return os.path.join(CURRENT_DIR, "hmm_metaphase_parameters.json")
    raise ValueError(f"Unknown model name: {model_name}")
