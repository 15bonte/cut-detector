import os


def get_model_path(model_name: str) -> None:
    """
    Returns absolute path to model file.
    """
    if model_name == "metaphase_model":
        return os.path.join(os.getcwd(), "metaphase_cnn", "metaphase_cnn.pt")
    if model_name == "hmm_metaphase_parameters":
        return os.path.join(os.getcwd(), "hmm_metaphase_parameters.json")
    raise ValueError(f"Unknown model name: {model_name}")
