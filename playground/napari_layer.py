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
    nbframes, height, width, _ = video.shape

    # Load mitosis tracks  # masques rajoutés qui suivent les cellules
    mitosis_tracks: list[MitosisTrack] = []
    for state_path in os.listdir(mitoses_path):
        with open(os.path.join(mitoses_path, state_path), "rb") as f:
            mitosis_track = pickle.load(f) # sauvegarder une instance de classe et la rechercher après
        mitosis_tracks.append(mitosis_track)

    # Colors list
    colors = np.array([[np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255)] for i in range(len(mitosis_tracks))], dtype=np.uint8)

    # Iterate over mitosis_tracks
    mask = np.zeros((nbframes,height,width,3), dtype=np.uint8)
    print("generate mask")
    for i,mitosis_track in enumerate(mitosis_tracks):
        _, mask_movie = mitosis_track.generate_video_movie(video)
        
        cell_indexes = np.where(mask_movie == 1)
        
        mask_movie = np.stack([mask_movie,mask_movie,mask_movie],axis=-1)
        
        mask_movie[cell_indexes] = colors[i]
        
        # Add mask_movie to viewer
        mask[mitosis_track.min_frame:mitosis_track.max_frame+1, mitosis_track.position.min_y:mitosis_track.position.max_y, mitosis_track.position.min_x:mitosis_track.position.max_x,:] = mask_movie
    
    viewer.add_image(mask, name="masks", rgb=True, opacity=0.4)
        

    # Display the Napari viewer
    napari.run()


if __name__ == "__main__":
    main(
        image_path = r"C:\Users\gwenaelle\git\cut-detector",
        mitoses_path = r"C:\Users\gwenaelle\git\cut-detector\mitoses"
    )
