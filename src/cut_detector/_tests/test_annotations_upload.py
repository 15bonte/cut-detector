from cut_detector.data.tools import get_data_path
from cut_detector.utils.tools import upload_annotations_folder


def test_annotations_upload():
    """Test manual CellCounter annotations upload."""

    annotations_folder = get_data_path("annotations")
    video_folder = get_data_path("videos")
    mitoses_folder = get_data_path("mitoses")

    _, not_detected = upload_annotations_folder(
        annotations_folder, video_folder, mitoses_folder, save=False
    )

    assert not_detected == 0
