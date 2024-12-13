import os
from typing import Optional
from scipy.stats import ttest_1samp
import numpy as np
import matplotlib.pyplot as plt
from napari import Viewer

from ..utils.parameters import Parameters
from ..utils.cell_track import CellTrack, generate_tracking_movie
from ..utils.tools import re_organize_channels
from ..utils.mitosis_track import MitosisTrack
from ..utils.mt_cut_detection.impossible_detection import (
    ImpossibleDetection,
)


def grayscale_to_rgb(grayscale_image, channel_axis, minimum_value=50):
    """Convert grayscale image to RGB image.

    Parameters
    ----------
    grayscale_image: np.ndarray
        Grayscale image. TYX.
    channel_axis: int
        Channel axis.

    Returns
    -------
    np.ndarray
        RGB image.
    """
    grayscale_image = grayscale_image.astype(np.float32)  # TYX
    grayscale_image = grayscale_image / grayscale_image.max() * 255
    grayscale_image = np.clip(grayscale_image, 0, 255)
    # Avoid values close to 0 as they are too dark
    grayscale_image[grayscale_image > 0] = (
        grayscale_image[grayscale_image > 0] / 255 * (255 - minimum_value)
        + minimum_value
    )
    grayscale_image = grayscale_image.astype(np.uint8)  # TYX

    indexes = np.unique(grayscale_image)
    fake_second_channel = np.copy(grayscale_image)
    fake_third_channel = np.copy(grayscale_image)

    random_colors = get_random_different_colors(len(indexes), nb_channels=2)
    for random_color, index in zip(random_colors, indexes):
        if index == 0:
            continue
        fake_second_channel[grayscale_image == index] = random_color[0]
        fake_third_channel[grayscale_image == index] = random_color[1]

    grayscale_image = np.stack(
        [
            grayscale_image,
            fake_second_channel,
            fake_third_channel,
        ],
        axis=-1,
    )  # TYXC

    # Match original image shape
    grayscale_image = np.moveaxis(grayscale_image, 3, channel_axis)  # TCYX
    return grayscale_image


def get_random_different_colors(
    nb_colors: int, minimum_value=50, nb_channels=3, interval=15
) -> np.ndarray:
    """Get random different colors.

    Parameters
    ----------
    nb_colors: int
        Number of colors.
    minimum_value: int
        Minimum value for colors to avoid dark colors. The default is 50.
    nb_channels: int
        Number of channels. The default is 3.
    interval: int
        Interval between colors. The default is 15.

    Returns
    -------
    np.ndarray
        Colors. Shape: (nb_colors, nb_channels).
    """
    colors = []
    for _ in range(nb_channels):
        colors_pick_list = np.array(range(minimum_value, 256))
        channel_colors = []
        for _ in range(nb_colors):
            if len(colors_pick_list) == 0:
                colors_pick_list = np.array(range(minimum_value, 256))
            random_color = np.random.choice(colors_pick_list)
            channel_colors.append(random_color)
            # Remove values close to random_color to avoid similar colors
            colors_pick_list = np.delete(
                colors_pick_list,
                np.where(np.abs(colors_pick_list - random_color) < interval),
            )
        colors.append(channel_colors)

    colors = np.array(colors).T.astype(np.uint8)
    return np.array(colors)


class ResultsSavingFactory:
    """Factory to save results.

    Parameters
    ----------
    max_frame: float
        Maximum frame to consider. The default is np.inf.
    """

    def __init__(
        self,
        params=Parameters(),
        max_frame=np.inf,
    ):
        self.params = params
        self.max_frame = max_frame  # for debug, 48*6 <-> 48h * 6 frames/hour

        self.first_cut_times: list[int] = []  # time in absolute time
        self.first_cut_times_gt: list[int] = []  # time in absolute time
        self.cut_differences: list[int] = []

        self.mitosis_results_summary: dict[int, int] = {}
        self.gt_mitosis_results_summary: dict[int, int] = {}

        self.p_value = None

    def _print_weird_mitoses(
        self, selected_tracks: list[MitosisTrack], min_acceptable_frame: int
    ) -> None:
        """Print useful information about weird mitoses.

        Parameters
        ----------
        selected_tracks: list[MitosisTrack]
            List of selected mitosis tracks.
        min_acceptable_frame: int
            Minimum acceptable frame for a cut to be considered as early.
        """
        ordered_tracks = [
            track
            for track in selected_tracks
            if track.key_events_frame["first_mt_cut"]
            - track.key_events_frame["no_mt_cut"]
            <= min_acceptable_frame
        ]
        ordered_tracks.sort(
            key=lambda x: x.key_events_frame["first_mt_cut"]
            - x.key_events_frame["no_mt_cut"]
        )

        print("Weird mitoses (early cut):")
        for mitosis_track in ordered_tracks:
            print("")
            print(
                f"Track: {mitosis_track.id}_{mitosis_track.mother_track_id}_to_{','.join(str(daughter) for daughter in mitosis_track.daughter_track_ids)}"
            )
            print(f"Key events frame: {mitosis_track.key_events_frame}")
            print(
                "video frame cut",
                mitosis_track.key_events_frame["first_mt_cut"]
                - mitosis_track.min_frame
                + 1,
            )
            if mitosis_track.gt_key_events_frame is not None:
                print(
                    f"GT key events frame: {mitosis_track.gt_key_events_frame}"
                )
                print(
                    "video frame cut",
                    mitosis_track.gt_key_events_frame["first_mt_cut"]
                    - mitosis_track.min_frame
                    + 1,
                )

    def update_cut_times(
        self,
        mitosis_tracks: list[MitosisTrack],
        verbose: bool,
        min_acceptable_frame=13,
    ) -> None:
        """Update cut times and summary.

        Parameters
        ----------
        mitosis_tracks: list[MitosisTrack]
            List of mitosis tracks.
        verbose: bool
            Whether to print details about weird mitoses.
        min_acceptable_frame: int
            Minimum acceptable frame for a cut to be considered as early.
        """

        selected_tracks: list[MitosisTrack] = []  # kept tracks

        for mitosis_track in mitosis_tracks:
            # Get first cut frame and start of cytokinesis frame
            cyto_frame = mitosis_track.key_events_frame["no_mt_cut"]
            cut_frame = mitosis_track.key_events_frame["first_mt_cut"]

            if cut_frame < 0 or cut_frame > self.max_frame:
                # No cut detected, for some reason
                cut_time = None
            else:
                # At least one cut was actually detected
                assert cut_frame >= cyto_frame  # should not be possible
                # Add the difference time to the list
                cut_time = (
                    cut_frame - cyto_frame
                ) * self.params.time_resolution
                self.first_cut_times.append(cut_time)
                selected_tracks.append(mitosis_track)

            # Update summary
            cut_id = min(cut_frame, 0)  # 0 if normal, negative otherwise

            if cut_id not in self.mitosis_results_summary:
                self.mitosis_results_summary[cut_id] = 0
            self.mitosis_results_summary[cut_id] += 1

            # Ignore if information is not there
            if (
                mitosis_track.gt_key_events_frame is None
                or "first_mt_cut" not in mitosis_track.gt_key_events_frame
            ):
                continue

            if (
                mitosis_track.gt_key_events_frame["first_mt_cut"]
                > self.max_frame
            ):
                # Ignore mitoses that are too late
                continue

            cut_time_gt = (
                mitosis_track.gt_key_events_frame["first_mt_cut"]
                - mitosis_track.gt_key_events_frame["no_mt_cut"]
            ) * self.params.time_resolution
            self.first_cut_times_gt.append(cut_time_gt)

            if cut_time is not None:
                self.cut_differences.append(cut_time - cut_time_gt)

            # Update ground truth summary
            if cut_id not in self.gt_mitosis_results_summary:
                self.gt_mitosis_results_summary[cut_id] = 0
            self.gt_mitosis_results_summary[cut_id] += 1

        # Display few details on mitoses with very early cut
        if verbose:
            self._print_weird_mitoses(selected_tracks, min_acceptable_frame)

    def perform_t_test(self, alpha=0.05) -> None:
        """Compute t-test of the differences, which is supposed to be 0.

        Parameters
        ----------
        alpha: float
            Significance level. The default is 0.05.
        """
        if len(self.cut_differences) == 0:
            return None

        t_result = ttest_1samp(self.cut_differences, 0)
        self.p_value = t_result.pvalue

        # Output the results
        print(f"P-value: {self.p_value}")

        # Check if the result is statistically significant (e.g., using a significance level of 0.05)
        if self.p_value < alpha:
            print(
                "Reject the null hypothesis: There is a statistically significant difference."
            )
        else:
            print(
                "Fail to reject the null hypothesis: There is no statistically significant difference."
            )

    def print_analysis_summary(self, mitosis_tracks) -> None:
        """Print analysis summary.

        Parameters
        ----------
        mitosis_tracks: list[MitosisTrack]
            List of mitosis tracks.
        """
        print(f"Among the {len(mitosis_tracks)} mitoses detected, there are:")
        for cut_id, count in self.mitosis_results_summary.items():
            print(
                f"    - {count} mitoses in category {ImpossibleDetection(cut_id).name}"
            )

        if len(self.first_cut_times_gt) == 0:
            return

        print(
            f"Among the {len(self.first_cut_times_gt)} mitoses annotated, there are:"
        )
        for cut_id, count in self.gt_mitosis_results_summary.items():
            print(
                f"    - {count} mitoses in category {ImpossibleDetection(cut_id).name}"
            )

    def box_plot_cut_differences(
        self, show: bool, save_dir: Optional[str]
    ) -> None:
        """Plot box plot of cut differences.

        Parameters
        ----------
        show: bool
            Whether to show the plot.
        save_dir: Optional[str]
            Directory to save the plot.
        """
        if len(self.cut_differences) == 0:
            return

        plt.figure()
        plt.boxplot(self.cut_differences)
        plt.title("Box plot of cut differences")
        plt.ylabel("Cut difference (min)")

        if show:
            plt.show()

        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, "first_mt_cut_differences.png"))

    def plot_cut_distributions(
        self,
        show: bool,
        save_dir: Optional[str],
    ) -> None:
        """Plot distribution of cut differences.

        Parameters
        ----------
        show: bool
            Whether to show the plot.
        save_dir: Optional[str]
            Directory to save the plot.
        """

        plt.figure()

        for cut_times, name in zip(
            [self.first_cut_times, self.first_cut_times_gt],
            ["Predicted", "Ground truth"],
        ):
            if len(cut_times) == 0:  # ignore empty gt list
                continue
            curve_values = []
            nb_values = len(cut_times)
            number_of_cut = 0
            for i in range(max(cut_times) + 1):
                count = cut_times.count(i)
                number_of_cut += count
                curve_values.append(number_of_cut / nb_values)
            plt.plot(curve_values, label=name)

        # Quartiles and median
        predicted_median = np.median(self.first_cut_times)
        gt_median = (
            np.median(self.first_cut_times_gt)
            if len(self.first_cut_times_gt)
            else None
        )
        predicted_q1 = np.percentile(self.first_cut_times, 25)
        predicted_q3 = np.percentile(self.first_cut_times, 75)
        plt.axvline(
            x=predicted_q1,
            color="r",
            linestyle="--",
            linewidth=0.5,
            label=f"Detected Q1: {predicted_q1}",
        )
        plt.axvline(
            x=predicted_median,
            color="r",
            linestyle="-",
            linewidth=0.5,
            label=f"Detected median: {predicted_median}"
            + (f"vs {gt_median} Ground truth median" if gt_median else ""),
        )
        plt.axvline(
            x=predicted_q3,
            color="r",
            linestyle="--",
            linewidth=0.5,
            label=f"Detected Q3: {predicted_q3}",
        )

        # add t test result to legend
        if self.p_value is not None:
            plt.legend(
                loc="lower right", title=f"t test p-value: {self.p_value:.2f}"
            )
        else:
            plt.legend(loc="lower right")

        plt.title(
            f"Proportion of MT cuts over time - {len(self.first_cut_times)} predicted mitoses"
            + (
                f"- {len(self.first_cut_times_gt)} ground truth mitoses"
                if len(self.first_cut_times_gt)
                else ""
            )
        )
        plt.xlabel("Time (min)")
        plt.ylabel("Proportion of first cut")

        if show:
            plt.show()

        if save_dir is not None:
            plt.savefig(
                os.path.join(save_dir, "first_mt_cut_distributions.png")
            )

    def save_csv_results(
        self,
        mitosis_tracks: list[MitosisTrack],
        mitosis_video_names: list[str],
        save_dir: Optional[str] = None,
    ) -> None:
        """Save results in CSV file.

        Parameters
        ----------
        mitosis_tracks: list[MitosisTrack]
            List of mitosis tracks.
        mitosis_video_names: list[str]
            List of mitosis video names.
        save_dir: Optional[str]
            Directory to save the results.
        """
        if save_dir is None:
            return

        csv_path = os.path.join(save_dir, "results.csv")

        column_names = [
            "video",
            "id",
            "mother track",
            "daughter track(s)",
            "metaphase frame - mitosis video",
            "cytokinesis frame - mitosis video",
            "first MT cut frame - mitosis video",
            "second MT cut frame - mitosis video",
            "metaphase frame",
            "cytokinesis frame",
            "first MT cut frame",
            "second MT cut frame",
            "position midbody x",
            "position midbody y",
            "first MT cut time",
            "second MT cut time",
        ]

        # Create CSV results file if it does not exist
        if not os.path.exists(csv_path):
            with open(csv_path, "w") as f:
                f.write(";".join(column_names) + "\n")
            f.close()

        # Store useful results in global results file
        with open(csv_path, "a") as f:
            for m_track, mitosis_video_name in zip(
                mitosis_tracks, mitosis_video_names
            ):
                f.write(f"{mitosis_video_name};")
                f.write(f"{m_track.id};")
                f.write(f"{m_track.mother_track_id};")
                # Daughter ids
                daughter_ids = ",".join(
                    [str(d) for d in m_track.daughter_track_ids]
                )
                f.write(f"{daughter_ids};")
                # Relative frames
                f.write(
                    f"{m_track.get_event_frame('metaphase', relative=True)};"
                )
                cytokinesis_frame = m_track.get_event_frame(
                    "no_mt_cut", relative=True
                )
                f.write(f"{cytokinesis_frame};")
                first_cut_frame = m_track.get_event_frame(
                    "first_mt_cut", relative=True
                )
                f.write(f"{first_cut_frame};")
                second_cut_frame = m_track.get_event_frame(
                    "second_mt_cut", relative=True
                )
                f.write(f"{second_cut_frame};")
                # Absolute frames
                f.write(
                    f"{m_track.get_event_frame('metaphase', relative=False)};"
                )
                f.write(
                    f"{m_track.get_event_frame('no_mt_cut', relative=False)};"
                )
                f.write(
                    f"{m_track.get_event_frame('first_mt_cut', relative=False)};"
                )
                f.write(
                    f"{m_track.get_event_frame('second_mt_cut', relative=False)};"
                )
                # Positions
                first_mb_position = m_track.get_first_mid_body_position()
                mb_x, mb_y = first_mb_position["x"], first_mb_position["y"]
                f.write(f"{mb_x};")
                f.write(f"{mb_y};")
                # Cut times
                first_cut_time = (
                    (first_cut_frame - cytokinesis_frame)
                    * self.params.time_resolution
                    if isinstance(first_cut_frame, int)
                    else ""
                )
                f.write(f"{first_cut_time};")
                second_cut_time = (
                    (second_cut_frame - cytokinesis_frame)
                    * self.params.time_resolution
                    if isinstance(second_cut_frame, int)
                    else ""
                )
                f.write(f"{second_cut_time};\n")
        f.close()

    def generate_napari_tracking_mask(
        self,
        mitosis_tracks: list[MitosisTrack],
        video: np.ndarray,
        viewer: Optional[Viewer] = None,
        segmentation_results: Optional[np.ndarray] = None,
        cell_tracks: Optional[list[CellTrack]] = None,
    ) -> None:
        """Generate napari tracking mask.

        Parameters
        ----------
        mitosis_tracks: list[MitosisTrack]
            List of mitosis tracks.
        video: np.ndarray
            Video to process. Any dimension order.
        viewer: napari.Viewer
            Napari viewer.
        segmentation_results: np.ndarray
            Cellpose results. TYX.
        """

        channel_axis = np.argmin(video.shape)
        # By default, Napari assumes that images with 3 dimensions in last
        # axis are RGB
        rgb = channel_axis == 3

        # Re-organize channels
        video_to_process = re_organize_channels(video)  # TYXC

        # Video parameters
        nb_frames, height, width, _ = video_to_process.shape

        # Colors list
        colors = get_random_different_colors(len(mitosis_tracks))

        # Iterate over mitosis_tracks
        mitoses_results = np.zeros(
            (nb_frames, height, width, 3), dtype=np.uint8
        )  # TYXC
        for idx, mitosis_track in enumerate(mitosis_tracks):
            if not mitosis_track.display():
                continue
            _, mask_movie = mitosis_track.generate_video_movie(
                video_to_process
            )
            cell_indexes = np.where(mask_movie == 1)
            mask_movie = np.stack(
                [mask_movie, mask_movie, mask_movie], axis=-1
            )
            mask_movie[cell_indexes] = colors[idx]
            initial_mask = mitoses_results[
                mitosis_track.min_frame : mitosis_track.max_frame + 1,
                mitosis_track.position.min_y : mitosis_track.position.max_y,
                mitosis_track.position.min_x : mitosis_track.position.max_x,
                :,
            ]
            # Avoid colors overlap
            mitoses_results[
                mitosis_track.min_frame : mitosis_track.max_frame + 1,
                mitosis_track.position.min_y : mitosis_track.position.max_y,
                mitosis_track.position.min_x : mitosis_track.position.max_x,
                :,
            ] = np.maximum(mask_movie, initial_mask)
        # Match original image shape
        mitoses_results = np.moveaxis(mitoses_results, 3, channel_axis)  # TCYX
        assert mitoses_results.shape == video.shape

        # Use point + text instead of red point for mid_body
        points = []
        features = {"category": []}
        text = {
            "string": "{category}",
            "size": 10,
            "color": "white",
            "translation": np.array([-30, 0]),
        }
        for mitosis_track in mitosis_tracks:
            if not mitosis_track.display():
                continue
            mid_body_legend = mitosis_track.get_mid_body_legend()
            for frame, frame_dict in mid_body_legend.items():
                single_layer_points = np.array(
                    [frame, frame_dict["y"], frame_dict["x"]]
                )
                if rgb:
                    points += [single_layer_points]
                    features["category"] += [frame_dict["category"]]
                else:
                    for idx in range(3):  # 3 channels
                        layer_points = np.insert(
                            single_layer_points, channel_axis, idx
                        )
                        points += [layer_points]
                        features["category"] += [frame_dict["category"]]
        features["category"] = np.array(features["category"])

        if viewer is None:
            return

        viewer.add_image(
            mitoses_results,
            name="Cell divisions",
            opacity=0.4,
            rgb=rgb,
            colormap=None if rgb else "inferno",
        )
        viewer.add_points(
            points,
            features=features,
            text=text,
            name="Mid-bodies",
        )

        if segmentation_results is not None:
            segmentation_results = grayscale_to_rgb(
                segmentation_results, channel_axis
            )
            assert segmentation_results.shape == video.shape

            viewer.add_image(
                segmentation_results,
                name="Segmentation",
                opacity=0.4,
                rgb=rgb,
                colormap=None if rgb else "inferno",
                visible=False,
            )

        if cell_tracks is not None:
            mitoses_results = generate_tracking_movie(
                cell_tracks, video_to_process
            )  # TYX

            mitoses_results = grayscale_to_rgb(mitoses_results, channel_axis)
            assert mitoses_results.shape == video.shape

            viewer.add_image(
                mitoses_results,
                name="Tracking",
                opacity=0.4,
                rgb=rgb,
                colormap=None if rgb else "inferno",
                visible=False,
            )
