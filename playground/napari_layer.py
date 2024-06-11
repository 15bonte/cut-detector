import os
from typing import Optional
import pickle
from skimage import io
import napari

from cut_detector.data.tools import get_data_path
from cut_detector.factories.results_saving_factory import ResultsSavingFactory


def main(
    image_path: Optional[str] = get_data_path("videos"),
    mitoses_path: Optional[str] = get_data_path("mitoses"),
):
    # Create a Napari viewer
    viewer = napari.Viewer()

    # Add video
    video = io.imread(os.path.join(image_path, "example_video.tif"))  # TYXC
    # Move axes to TCYX to match napari
    viewer_video = video.transpose(0, 3, 1, 2)  # TCYX
    viewer.add_image(viewer_video, name="video", rgb=False)

    # Load mitosis tracks  # masques rajoutés qui suivent les cellules
    mitosis_tracks = []
    for state_path in os.listdir(mitoses_path):
        with open(os.path.join(mitoses_path, state_path), "rb") as f:
            mitosis_track = pickle.load(f)
        mitosis_tracks.append(mitosis_track)

    ResultsSavingFactory().generate_napari_tracking_mask(
        mitosis_tracks, viewer_video, viewer
    )

    napari.run()


if __name__ == "__main__":
    main()
