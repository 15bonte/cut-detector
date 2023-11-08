import os
import urllib.request

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_data_path(data_name: str) -> None:
    """
    Returns absolute path to model file.
    """
    sub_folder_to_create = data_name
    if data_name == "mitoses":
        files = ["example_video_mitosis_0_4_to_0.bin"]
    elif data_name == "models":
        files = ["example_video_model.xml"]
    elif data_name == "tracks":
        sub_folder_to_create = f"{data_name}/example_video"
        files = [
            "example_video/track_0.bin",
            "example_video/track_1.bin",
            "example_video/track_2.bin",
            "example_video/track_3.bin",
            "example_video/track_4.bin",
        ]
    elif data_name == "results":
        files = []  # no files to download here
    elif data_name == "videos":
        files = ["example_video.tif"]
    else:
        raise ValueError(f"Unknown data name: {data_name}")

    local_path = os.path.join(CURRENT_DIR, data_name)

    if sub_folder_to_create is not None:
        os.makedirs(os.path.join(CURRENT_DIR, sub_folder_to_create), exist_ok=True)

    for file in files:
        file_local_path = os.path.join(local_path, file)
        if not os.path.exists(file_local_path):
            print(f"Downloading data {data_name}...")
            urllib.request.urlretrieve(
                f"https://raw.githubusercontent.com//15bonte/cut-detector-models/main/data/{data_name}/{file}",
                file_local_path,
            )

    return local_path
