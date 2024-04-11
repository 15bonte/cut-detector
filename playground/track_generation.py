from skimage import io

from cut_detector.widget_functions.mitosis_track_generation import (
    perform_mitosis_track_generation,
)


def main(video_path, xml_video_dir):
    # Add video
    video = io.imread(video_path)  # TYXC
    video_name = video_path.split("\\")[-1].split(".")[0]

    perform_mitosis_track_generation(video, video_name, xml_video_dir)


if __name__ == "__main__":
    VIDEO_PATH = r"C:\Users\thoma\data\Data Pasteur cep55\videos\20231019-t1_siCep55-50-1.tif"
    XML_VIDEO_DIR = r"C:\Users\thoma\data\Cut Detector Debug\xml"
    main(VIDEO_PATH, XML_VIDEO_DIR)
