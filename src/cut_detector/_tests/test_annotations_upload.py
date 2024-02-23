import os
from cut_detector.data.tools import get_data_path
from cut_detector.utils.tools import upload_annotations


def test_annotations_upload(
    annotations_folder=get_data_path("annotations"),
    video_folder=get_data_path("videos"),
    mitoses_folder=get_data_path("mitoses"),
):
    # Initialize variables
    all_detected, all_not_detected = 0, 0

    video_paths = os.listdir(video_folder)
    for video_idx, video in enumerate(video_paths):
        video_progress = f"{video_idx + 1}/{len(video_paths)}"
        print(f"\nVideo {video_progress}...")
        # Perform mid-body detection evaluation
        detected, not_detected = upload_annotations(
            annotations_folder,
            os.path.join(video_folder, video),
            mitoses_folder,
            update_mitoses=False,
        )
        all_detected += detected
        all_not_detected += not_detected

    assert all_not_detected == 0
