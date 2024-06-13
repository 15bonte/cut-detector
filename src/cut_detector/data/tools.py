import os
import urllib.request

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_data_path(data_name: str) -> None:
    """
    Returns absolute path to model file.
    """
    sub_folder_to_create = data_name
    if data_name == "mitoses":
        files = ["example_video_mitosis_0_0_to_5.bin"]
    elif data_name == "tracks":
        sub_folder_to_create = f"{data_name}/example_video"
        files = [
            f"example_video/track_{track_id}.bin"
            for track_id in [0, 1, 2, 3, 5]
        ]
    elif data_name == "results":
        files = []  # no files to download here
    elif data_name == "videos":
        files = ["example_video.tif"]
    elif data_name == "mitosis_movies":
        files = ["example_video_mitosis_0_0_to_5.tiff"]
    elif data_name == "mid_bodies":
        files = []  # no files to download here
    elif data_name == "mid_bodies_tests":
        files = []  # no files to download here
    elif data_name == "segmentation_results":
        files = ["example_video.bin"]
    elif data_name == "spots":
        sub_folder_to_create = f"{data_name}/example_video"
        files = [f"example_video/spot_{spot_id}.bin" for spot_id in range(236)]
    elif data_name == "annotations":
        sub_folder_to_create = f"{data_name}/example_video"
        files = [
            "example_video/CellCounter_example_video_mitosis_0_0_to_5.xml"
        ]
    else:
        raise ValueError(f"Unknown data name: {data_name}")

    local_path = os.path.join(CURRENT_DIR, data_name)
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
