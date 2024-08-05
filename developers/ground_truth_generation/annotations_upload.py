from cut_detector.utils.tools import upload_annotations_folder

if __name__ == "__main__":

    ANNOTATIONS_FOLDER = ""
    VIDEO_FOLDER = ""
    MITOSES_FOLDER = ""

    upload_annotations_folder(
        ANNOTATIONS_FOLDER, VIDEO_FOLDER, MITOSES_FOLDER, save=True
    )
