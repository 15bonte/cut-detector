import os
from typing import Optional
from scipy.stats import ttest_1samp
import numpy as np
import matplotlib.pyplot as plt

from ..utils.mitosis_track import MitosisTrack
from ..utils.bridges_classification.impossible_detection import (
    ImpossibleDetection,
)


class ResultsSavingFactory:
    """Factory to save results."""

    def __init__(
        self,
        time_resolution: Optional[int] = 10,
        max_frame: Optional[int] = np.inf,
    ):
        self.time_resolution = time_resolution
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
        """
        Print useful information about weird mitoses.
        """
        ordered_tracks = [
            track
            for track in selected_tracks
            if track.key_events_frame["first_mt_cut"]
            - track.key_events_frame["cytokinesis"]
            <= min_acceptable_frame
        ]
        ordered_tracks.sort(
            key=lambda x: x.key_events_frame["first_mt_cut"]
            - x.key_events_frame["cytokinesis"]
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
        """Update cut times and summary."""

        selected_tracks: list[MitosisTrack] = []  # kept tracks

        for mitosis_track in mitosis_tracks:
            # Get first cut frame and start of cytokinesis frame
            cyto_frame = mitosis_track.key_events_frame["cytokinesis"]
            cut_frame = mitosis_track.key_events_frame["first_mt_cut"]

            if cut_frame < 0 or cut_frame > self.max_frame:
                # No cut detected, for some reason
                cut_time = None
            else:
                # At least one cut was actually detected
                assert cut_frame >= cyto_frame  # should not be possible
                # Add the difference time to the list
                cut_time = (cut_frame - cyto_frame) * self.time_resolution
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
                - mitosis_track.gt_key_events_frame["cytokinesis"]
            ) * self.time_resolution
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
        """
        Compute t-test of the differences, which is supposed to be 0.
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
        """Print analysis summary."""
        print(
            f"\n Among the {len(mitosis_tracks)} mitoses detected, there are:"
        )
        for cut_id, count in self.mitosis_results_summary.items():
            print(
                f"    - {count} mitoses in category {ImpossibleDetection(cut_id).name}"
            )

        if len(self.first_cut_times_gt) == 0:
            return
        print(
            f"\n Among the {len(self.first_cut_times_gt)} mitoses annotated, there are:"
        )
        for cut_id, count in self.gt_mitosis_results_summary.items():
            print(
                f"    - {count} mitoses in category {ImpossibleDetection(cut_id).name}"
            )

    def box_plot_cut_differences(
        self, show: bool, save_dir: Optional[str]
    ) -> None:
        """
        Plot box plot of cut differences
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
        """
        Plot distribution of cut differences
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
        save_dir: Optional[str] = None,
    ) -> None:
        """Save results in CSV file."""
        if save_dir is None:
            return

        csv_path = os.path.join(save_dir, "results.csv")
        # Create CSV results file if it does not exist
        if not os.path.exists(csv_path):
            with open(csv_path, "w") as f:
                f.write(
                    "id;mother track;daughter track(s);metaphase frame;cytokinesis frame;first MT cut frame;second MT cut frame;first MT cut time;second MT cut time\n"
                )
            f.close()

        # Store useful results in global results file
        with open(csv_path, "a") as f:
            for mitosis_track in mitosis_tracks:
                f.write(f"{mitosis_track.id};")
                f.write(f"{mitosis_track.mother_track_id};")
                daughter_ids = ",".join(
                    [str(d) for d in mitosis_track.daughter_track_ids]
                )
                f.write(f"{daughter_ids};")
                f.write(f"{mitosis_track.key_events_frame['metaphase']};")
                cytokinesis_frame = mitosis_track.key_events_frame[
                    "cytokinesis"
                ]
                f.write(f"{cytokinesis_frame};")
                first_cut_frame = mitosis_track.key_events_frame[
                    "first_mt_cut"
                ]
                f.write(f"{first_cut_frame};")
                second_cut_frame = mitosis_track.key_events_frame[
                    "second_mt_cut"
                ]
                f.write(f"{second_cut_frame};")
                first_cut_time = (
                    (first_cut_frame - cytokinesis_frame)
                    * self.time_resolution
                    if first_cut_frame >= 0
                    else ""
                )
                f.write(f"{first_cut_time};")
                second_cut_time = (
                    (second_cut_frame - cytokinesis_frame)
                    * self.time_resolution
                    if second_cut_frame >= 0
                    else ""
                )
                f.write(f"{second_cut_time};\n")
        f.close()
