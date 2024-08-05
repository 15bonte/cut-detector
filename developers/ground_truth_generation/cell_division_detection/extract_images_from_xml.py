import os
import random
import xml.etree.ElementTree as ET
from bigfish import stack


def extract_images(
    xml_model_path: str,
    video_path: str,
    save_dir: str,
    ratio_to_extract: float,
    circularity_threshold: float,
) -> None:
    """Extract images from a video based on a CellCounter xml file.

    Parameters
    ----------
    xml_model_path : str
        Path to the xml file.
    video_path : str
        Path to the video.
    save_dir : str
        Path to the folder where to save the images.
    ratio_to_extract : float
        Ratio of images to extract.
    circularity_threshold : float
        Circularity threshold to extract images.

    """
    root = ET.parse(xml_model_path).getroot()

    # Create save directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Read image
    raw_video = stack.read_image(video_path, sanity_check=False)  # T, H, W, C
    video_name = os.path.basename(video_path).split(".")[0]

    for type_tag in root.findall("Model/AllSpots/SpotsInFrame/Spot"):
        raw_positions = type_tag.text.split(" ")
        frame = int(type_tag.get("FRAME"))

        # Get min and max positions
        rel_positions_x = [
            float(raw_positions[i]) for i in range(0, len(raw_positions), 2)
        ]
        rel_positions_y = [
            float(raw_positions[i]) for i in range(1, len(raw_positions), 2)
        ]
        rel_min_x, rel_max_x = min(rel_positions_x), max(rel_positions_x)
        rel_min_y, rel_max_y = min(rel_positions_y), max(rel_positions_y)

        spot_position_x, spot_position_y = (
            float(type_tag.get("POSITION_X")),
            float(type_tag.get("POSITION_Y")),
        )
        abs_min_x, abs_max_x = int(spot_position_x + rel_min_x), int(
            spot_position_x + rel_max_x
        )
        abs_min_y, abs_max_y = int(spot_position_y + rel_min_y), int(
            spot_position_y + rel_max_y
        )

        abs_min_x, abs_max_x = max(abs_min_x, 0), min(
            abs_max_x, raw_video.shape[2]
        )
        abs_min_y, abs_max_y = max(abs_min_y, 0), min(
            abs_max_y, raw_video.shape[1]
        )

        nucleus = raw_video[frame, abs_min_y:abs_max_y, abs_min_x:abs_max_x, :]

        circularity = float(type_tag.get("CIRCULARITY"))

        if (
            circularity
            > circularity_threshold  # Extract rounded nuclei to have metaphase cells
            or random.random() < ratio_to_extract
            or frame
            > raw_video.shape[0]
            * 0.9  # Extract all last 10% of frames to have dead cells
        ):
            # Generate random index number
            random_index = random.randint(0, 100)
            stack.save_image(
                nucleus,
                f"{save_dir}/{video_name}_nucleus_{random_index}_{frame}.tif",
            )


if __name__ == "__main__":

    XML_MODEL_PATH = ""
    VIDEO_PATH = ""
    SAVE_DIR = ""

    RATIO_TO_EXTRACT = 0.01
    CIRCULARITY_THRESHOLD = 0.8

    extract_images(
        XML_MODEL_PATH,
        VIDEO_PATH,
        SAVE_DIR,
        RATIO_TO_EXTRACT,
        CIRCULARITY_THRESHOLD,
    )
