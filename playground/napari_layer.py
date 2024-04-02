import os
from typing import Optional
from skimage import io 
import numpy as np
import napari
import pickle
from cut_detector.data.tools import get_data_path
from cut_detector.utils.mitosis_track import MitosisTrack
from cut_detector.factories.results_saving_factory import ResultsSavingFactory


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
    nbframes, height, width, _ = video.shape

    # Load mitosis tracks  # masques rajoutés qui suivent les cellules
    mitosis_tracks: list[MitosisTrack] = []
    for state_path in os.listdir(mitoses_path):
        with open(os.path.join(mitoses_path, state_path), "rb") as f:
            mitosis_track: MitosisTrack = pickle.load(f) # sauvegarder une instance de classe et la rechercher après
            mitosis_track.adapt_deprecated_attributes()
        mitosis_tracks.append(mitosis_track)
  
    # TODO: move everything to this function:
    mask= ResultsSavingFactory().generate_napari_tracking_mask(mitosis_tracks, video)
    
    viewer.add_image(mask, name="masks", rgb=True, opacity=0.4)
    
     # Display the Napari viewer
    napari.run()



if __name__ == "__main__":
    main(
        image_path = r"C:\Users\camca\Documents\video_exemple",
        mitoses_path = r"C:\Users\camca\Documents\video_exemple\mitoses"
    )
