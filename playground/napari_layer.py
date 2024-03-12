import os
from typing import Optional
from skimage import io 
import numpy as np
import napari
import pickle

from cut_detector.data.tools import get_data_path
from cut_detector.utils.mitosis_track import MitosisTrack


def main(
    image_path: Optional[str] = get_data_path("videos"),
    mitoses_path: Optional[str] = get_data_path("mitoses"),
):
    # Create a Napari viewer
    viewer = napari.Viewer()

    # Add video
    video = io.imread(os.path.join(image_path, "example_video.tif"))  # TYXC
    viewer.add_image(video[..., 0].squeeze(), name="micro-tubules")
    viewer.add_image(video[..., 1].squeeze(), name="mid-body")
    viewer.add_image(video[..., 2].squeeze(), name="phase contrast")

    # Create a blue rectangle
    rectangle_width = 50
    rectangle_height = 30
    rectangle_color = [0, 0, 1]  # Blue color in RGB

    # Define the rectangle vertices
    vertices = np.array(
        [
            [0, 0],
            [rectangle_width, 0],
            [rectangle_width, rectangle_height],
            [0, rectangle_height],
        ]
    )

    # Create a rectangle layer
    viewer.add_shapes(
        data=vertices,
        shape_type="rectangle",
        edge_color="transparent",
        face_color=rectangle_color,
    )

    # Load mitosis tracks
    mitosis_tracks: list[MitosisTrack] = []
    for state_path in os.listdir(mitoses_path):
        with open(os.path.join(mitoses_path, state_path), "rb") as f:
            mitosis_track = pickle.load(f)
        mitosis_tracks.append(mitosis_track)

    # Iterate over mitosis_tracks
    for mitosis_track in mitosis_tracks:
        _, mask_movie = mitosis_track.generate_video_movie(video)
        # Add mask_movie to viewer
        # TODO

    # Display the Napari viewer
    napari.run()


if __name__ == "__main__":
    main()
