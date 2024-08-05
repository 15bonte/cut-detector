import os
import xmltodict

from cut_detector.constants.annotations import NAMES_DICTIONARY
from cut_detector.utils.tools import get_video_path


def check_file(xml_folder: str, mitoses_path: str) -> None:
    """Check the xml files in the given folder.

    Parameters
    ----------
    xml_folder : str
        Path to the folder containing xml CellCounter files.
    mitoses_path : str
        Path to the mitoses folder.
    """
    for xml_name in os.listdir(xml_folder):
        print(f"Processing {xml_name}...")
        xml_path = os.path.join(xml_folder, xml_name)

        # Read useful information from xml file
        with open(xml_path) as fd:
            doc = xmltodict.parse(fd.read())

        # Try to find video
        video_name = doc["CellCounter_Marker_File"]["Image_Properties"][
            "Image_Filename"
        ]
        video_path = get_video_path(video_name, mitoses_path)
        if video_path is None:
            raise FileNotFoundError(f"Video {video_name} not found.")

        # Check ids
        for i, type_data in enumerate(
            doc["CellCounter_Marker_File"]["Marker_Data"]["Marker_Type"]
        ):
            assert i == int(type_data["Type"]) - 1  # order should be kept
            if (
                "Name" not in type_data
                or type_data["Name"] not in NAMES_DICTIONARY
            ):
                old_name = type_data["Name"] if "Name" in type_data else ""
                new_name = list(NAMES_DICTIONARY.keys())[
                    list(NAMES_DICTIONARY.values()).index(i)
                ]
                print(
                    f"Class '{old_name}' not found, modified to '{new_name}'."
                )
                # Update accordingly
                doc["CellCounter_Marker_File"]["Marker_Data"]["Marker_Type"][
                    i
                ]["Name"] = new_name
            else:
                assert NAMES_DICTIONARY[type_data.Name] == i

        with open(xml_path, "w") as result_file:
            result_file.write(xmltodict.unparse(doc, pretty=True))


if __name__ == "__main__":
    XML_FOLDER = ""
    MITOSES_PATH = ""

    check_file(XML_FOLDER, MITOSES_PATH)
