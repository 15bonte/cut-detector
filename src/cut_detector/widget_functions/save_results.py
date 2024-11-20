import os
from typing import Optional

import numpy as np
import h5py

from ..utils.mt_cut_detection.impossible_detection import (
    ImpossibleDetection,
)
from ..utils.parameters import Parameters
from ..utils.image_tools import resize_image
from ..utils.cell_track import CellTrack
from ..factories.results_saving_factory import ResultsSavingFactory
from ..utils.mitosis_track import MitosisTrack


def perform_results_saving(
    exported_mitoses_dir: str,
    show: bool = False,
    save_dir: Optional[str] = None,
    verbose: bool = False,
    video: Optional[np.ndarray] = None,
    viewer: Optional["napari.Viewer"] = None,
    segmentation_results: Optional[np.ndarray] = None,
    cell_tracks: Optional[list[CellTrack]] = None,
    params=Parameters(),
) -> None:
    """Perform a series of tests, prints and plots following process.

    Parameters
    ----------
    video : np.ndarray
        Video. TYXC.
    exported_mitoses_dir : str
        Directory where mitosis tracks are saved.
    show : bool, optional
        Show plots, by default False.
    save_dir : Optional[str], optional
        Directory where to save results, by default None.
    verbose : bool, optional
        Verbose, by default False.
    video : Optional[np.ndarray], optional
        Video, by default None. Any dimension order.
    viewer : Optional["napari.Viewer"], optional
        Viewer, by default None.
    segmentation_results : Optional[np.ndarray], optional
        Segmentation results, by default None. TYX.
    """
    print("\n### RESULTS SAVING ###")

    # Create save_dir if specified and it does not exist
    if save_dir is not None and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    mitosis_tracks: list[MitosisTrack] = []
    mitosis_video_names: list[str] = []
    # Iterate over "bin" files in exported_mitoses_dir
    for state_path in os.listdir(exported_mitoses_dir):
        with open(os.path.join(exported_mitoses_dir, state_path), "rb") as f:
            mitosis_track = MitosisTrack.load(f)
            mitosis_tracks.append(mitosis_track)
            mitosis_video_names.append(state_path.split("_mitosis_")[0])

    # Define lists and dictionaries to store results
    results_saving_factory = ResultsSavingFactory(params=params)
    results_saving_factory.update_cut_times(mitosis_tracks, verbose)

    # Protect against no detection
    if len(results_saving_factory.first_cut_times) == 0:
        print("No mitosis detected.")
        return

    # Perform a series of tests, prints and plots
    results_saving_factory.perform_t_test()
    results_saving_factory.print_analysis_summary(mitosis_tracks)
    results_saving_factory.save_csv_results(
        mitosis_tracks, mitosis_video_names, save_dir
    )
    results_saving_factory.box_plot_cut_differences(show, save_dir)
    results_saving_factory.plot_cut_distributions(show, save_dir)

    # Display in napari
    if video is not None:
        print("\nDisplaying results in Napari.")
        results_saving_factory.generate_napari_tracking_mask(
            mitosis_tracks,
            video,
            viewer,
            segmentation_results=segmentation_results,
            cell_tracks=cell_tracks,
        )


def save_galleries(
    video,
    video_name,
    exported_mitoses_dir: str,
    save_dir,
    time=80,
    width=600,
    height=600,
) -> None:
    """Save impossible detections galleries.

    Parameters
    ----------
    video : np.ndarray
        Video. TYXC.
    video_name : str
        Video name.
    exported_mitoses_dir : str
        Directory where mitosis tracks are saved.
    save_dir : str
        Directory where to save results.
    """
    mitosis_tracks: list[MitosisTrack] = []
    # Iterate over "bin" files in exported_mitoses_dir
    for state_path in os.listdir(exported_mitoses_dir):
        # Ignore if not for current video
        if video_name not in state_path:
            continue
        # Load mitosis track
        with open(os.path.join(exported_mitoses_dir, state_path), "rb") as f:
            mitosis_track = MitosisTrack.load(f)

        # Add mitosis track to list
        mitosis_tracks.append(mitosis_track)

    # Classify mitosis tracks depending on detection status
    classified_mitosis_tracks: dict[str, list[MitosisTrack]] = {}
    for mitosis_track in mitosis_tracks:
        cut_frame = mitosis_track.key_events_frame["first_mt_cut"]
        if cut_frame >= 0:  # do not save normal mitoses
            continue
        category = ImpossibleDetection(cut_frame).name
        if category not in classified_mitosis_tracks:
            classified_mitosis_tracks[category] = []
        classified_mitosis_tracks[category].append(mitosis_track)

    # Create galleries
    for category, classified_tracks in classified_mitosis_tracks.items():
        gallery_path = os.path.join(
            save_dir, f"gallery_{video_name}_{category}.h5"
        )
        with h5py.File(gallery_path, "w") as h5file:

            images_dataset = h5file.create_dataset(
                "images",
                (len(classified_tracks), time, 4, height, width),  # BTCYX
                chunks=(1, time, 4, height, width),
                dtype=np.uint16,
            )

            for idx, mitosis_track in enumerate(classified_tracks):
                images_dataset.attrs[str(idx)] = mitosis_track.get_file_name(
                    video_name
                )
                mitosis_movie, mask_movie = mitosis_track.generate_video_movie(
                    video
                )  # TYXC
                final_mitosis_movie = mitosis_track.add_mid_body_movie(
                    mitosis_movie, mask_movie
                )  # TYX C=C+1
                movie = np.moveaxis(final_mitosis_movie, 3, 1)  # TCYX

                movie_to_save = []
                for frame in range(min(movie.shape[0], time)):
                    frame_to_save = resize_image(
                        movie[frame], (4, height, width)
                    )  # CYX
                    frame_to_save = np.flip(
                        frame_to_save, axis=1
                    )  # h5 has origin on bottom left
                    movie_to_save.append(frame_to_save)
                movie_to_save = np.stack(movie_to_save)  # TCYX

                if movie_to_save.shape[0] < time:
                    movie_to_save = np.pad(
                        movie_to_save,
                        (
                            (0, time - movie_to_save.shape[0]),
                            (0, 0),
                            (0, 0),
                            (0, 0),
                        ),
                        mode="constant",
                        constant_values=0,
                    )

                assert movie_to_save.shape == (time, 4, height, width)
                images_dataset[idx] = movie_to_save
