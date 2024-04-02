import os
import pickle
from typing import Optional

from ..factories.results_saving_factory import ResultsSavingFactory
from ..utils.mitosis_track import MitosisTrack


def perform_results_saving(
    exported_mitoses_dir: str,
    show: bool = False,
    save_dir: Optional[str] = None,
    verbose: bool = False,
) -> None:
    """
    Perform a series of tests, prints and plots following process.
    """
    # Create save_dir if specified and it does not exist
    if save_dir is not None and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    mitosis_tracks: list[MitosisTrack] = []
    # Iterate over "bin" files in exported_mitoses_dir
    for state_path in os.listdir(exported_mitoses_dir):
        with open(os.path.join(exported_mitoses_dir, state_path), "rb") as f:
            mitosis_track: MitosisTrack = pickle.load(f)
            mitosis_track.adapt_deprecated_attributes()
            mitosis_tracks.append(mitosis_track)

    # Define lists and dictionaries to store results
    results_saving_factory = ResultsSavingFactory()
    results_saving_factory.update_cut_times(mitosis_tracks, verbose)

    # Protect against no detection
    if len(results_saving_factory.first_cut_times) == 0:
        print("No mitosis detected.")
        return

    # Perform a series of tests, prints and plots
    results_saving_factory.perform_t_test()
    results_saving_factory.print_analysis_summary(mitosis_tracks)
    results_saving_factory.save_csv_results(mitosis_tracks, save_dir)
    results_saving_factory.box_plot_cut_differences(show, save_dir)
    results_saving_factory.plot_cut_distributions(show, save_dir)
