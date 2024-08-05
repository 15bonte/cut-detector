import os
import xmltodict
from bigfish import stack
from munch import Munch
from cnn_framework.utils.tools import save_tiff

from cut_detector.factories.mt_cut_detection_factory import (
    MtCutDetectionFactory,
)
from cut_detector.utils.image_tools import (
    cell_counter_frame_to_video_frame,
    smart_cropping,
)
from cut_detector.constants.annotations import NAMES_DICTIONARY
from cut_detector.utils.tools import get_video_path


def extract_bridge_images(
    xml_folders, division_movies_path, save_dir, classes_to_keep
) -> None:
    """Extract bridge images from xml files.

    Parameters
    ----------
    xml_folders : list[str]
        List of paths to the folders containing xml CellCounter files.
    division_movies_path : str
        Path to the division movies folder.
    save_dir : str
        Path to the folder where to save the images.
    """
    margin = MtCutDetectionFactory().margin

    # Create save_dir if not exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for xml_folder in xml_folders:
        for xml_name in os.listdir(xml_folder):
            print(f"Processing {xml_name}...")
            xml_path = os.path.join(xml_folder, xml_name)

            # Read useful information from xml file
            with open(xml_path) as fd:
                doc = Munch.fromDict(xmltodict.parse(fd.read()))

            # Read video
            video_name = (
                doc.CellCounter_Marker_File.Image_Properties.Image_Filename
            )
            video_path = get_video_path(video_name, division_movies_path)
            if video_path is None:
                print(f"Video {video_name} not found. Skipping.")
                continue

            raw_video = stack.read_image(
                video_path, sanity_check=False
            )  # TYXC
            video_save_name = os.path.basename(video_path).split(".")[0]
            # Switch channels to beginning
            raw_video = raw_video.transpose(3, 0, 1, 2)  # CYXW
            nb_channels = raw_video.shape[0]

            classes_max_frames = [-1] * 5
            for i, type_data in enumerate(
                doc.CellCounter_Marker_File.Marker_Data.Marker_Type
            ):
                assert i == int(type_data.Type) - 1  # order should be kept
                class_index = NAMES_DICTIONARY[type_data.Name]
                # Ignore if no data
                if "Marker" not in type_data:
                    continue
                # Ignore if current class as to be ignored
                if class_index not in classes_to_keep:
                    continue
                markers = type_data.Marker
                if not isinstance(markers, list):
                    markers = [markers]
                for marker in markers:
                    x_pos, y_pos, frame = (
                        int(marker.MarkerX),
                        int(marker.MarkerY),
                        cell_counter_frame_to_video_frame(
                            int(marker.MarkerZ), nb_channels
                        ),
                    )
                    # Update max frame
                    classes_max_frames[(i + 1) % 5] = max(
                        classes_max_frames[(i + 1) % 5], frame
                    )
                    assert frame > classes_max_frames[(i + 1) % 5 - 1]
                    # Extract image
                    frame_image = raw_video[:, frame, :, :].squeeze()  # CYX
                    crop = smart_cropping(
                        frame_image, margin, x_pos, y_pos, pad=True
                    )  # CYX
                    # Define name
                    image_name = f"{video_save_name}_{str(frame).zfill(3)}_c{class_index}.tif"
                    save_tiff(
                        crop,
                        os.path.join(save_dir, image_name),
                        original_order="CYX",
                    )


if __name__ == "__main__":
    XML_FOLDERS = [""]
    DIVISION_MOVIES_PATH = ""
    SAVE_DIR = ""

    CLASSES_TO_KEEP = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    extract_bridge_images(
        XML_FOLDERS, DIVISION_MOVIES_PATH, SAVE_DIR, CLASSES_TO_KEEP
    )
