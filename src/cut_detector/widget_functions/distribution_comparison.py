import os
import pandas as pd
from plotly.subplots import make_subplots

from ..utils.division import (
    Division,
)
from ..utils.statistical_tests import (
    create_plot,
    mannwhitneyu_difference,
    mannwhitneyu_equivalence,
    wilcoxon_equivalence,
    wilcoxon_difference,
)

class VerbosePrinter:
    def __init__(self, verbose):
        self.verbose = verbose
    
    def print(self, message):
        if self.verbose:
            print(message)

def perform_distribution_comparison(
    matched_csv,
    manual_csv,
    save_folder,
    max_cytokinesis_frame,
    conditions_vs_manual,
    conditions_vs_control,
    deltas=[10, 20, 30, 40],
    mode="cumulative",
    show=False,
    save_fig=False,
    verbose=False,
)-> None:
    """
    Compare distributions of Cut Detector and manual annotations using statistical tests.

    Parameters
    ----------
    matched_csv : str
        Path to the matched CSV file.
    manual_csv : str
        Path to the manual CSV file.
    save_folder : str
        Path to save the plots and results.
    max_cytokinesis_frame : int
        Ignore divisions after this frame.
    conditions_vs_manual : list of str
        List of conditions to compare with manual annotations.
    conditions_vs_control : list of str
        List of conditions to compare to control.
    deltas : list of int, optional
        List of deltas to consider. Default is [10, 20, 30, 40].
    mode : str, optional
        Mode for the plot, either "cumulative" or "box". Default is "cumulative".
    show : bool, optional
        Whether to show the plots. Default is False.
    save_fig : bool, optional
        Whether to save the plots. Default is False.
    verbose : bool, optional
        Whether to print verbose messages. Default is False.
    """
    hour_date_string = pd.Timestamp.now().strftime("%Y-%m-%d %H-%M-%S")

    # Remove empty strings from conditions_vs_manual and conditions_vs_control
    conditions_vs_manual = [condition for condition in conditions_vs_manual if condition]
    conditions_vs_control = [condition for condition in conditions_vs_control if condition]

    divisions = []
    printer = VerbosePrinter(verbose)

    # Read matched csv
    df = pd.read_csv(matched_csv, delimiter=";")
    sheet_data = [df[col].tolist() for col in df.columns]
    for idx in range(len(sheet_data[0])):
        division_data = [col[idx] for col in sheet_data]
        if len(division_data[11]) > 3:  # impossible detection
            continue
        division = Division(division_data)
        divisions.append(division)

    # Read manual csv
    df = pd.read_csv(manual_csv, delimiter=";")
    sheet_data = [df[col].tolist() for col in df.columns]
    for idx in range(len(sheet_data[0])):
        division = Division([col[idx] for col in sheet_data])
        if not division.exists_in(divisions):
            divisions.append(division)

    divisions = [
        division
        for division in divisions
        if division.cytokinesis <= max_cytokinesis_frame
        and division.cd_cytokinesis <= max_cytokinesis_frame
    ]

    if save_folder:
        Division.generate_csv_summary(divisions, save_folder, hour_date_string)

    results_header = ["EXP", "Comparison", "Condition 1", "Condition 2", "Delta", "Test", "p-value", "Result"]
    results = []

    experiments_set = set(sheet_data[0] + ["EXP"])  # EXP gathers all

    # Compare Cut Detector and manual annotations
    for condition in conditions_vs_manual:

        for exp in experiments_set:

            printer.print(f"\n### Condition {condition} {exp} - Cut Detector vs Manual ###")
            only_cd, only_manual, both = Division.split_divisions(
                divisions, condition, exp
            )
            numbers_summary = f"\nCondition {condition} - Same: {len(both)}, FP: {len(only_cd)}, FN: {len(only_manual)}"
            printer.print(numbers_summary)

            for delta in deltas:

                printer.print(f"\n### Delta {delta} ###")

                # Same
                printer.print("# Same #")
                cd_cuts = Division.get_cuts(both, div_type="cd")
                manual_cuts = Division.get_cuts(both, div_type="manual")
                title_1, p_value = wilcoxon_equivalence(cd_cuts, manual_cuts, delta=delta, verbose=verbose)
                fig_1 = create_plot(
                    [cd_cuts, manual_cuts],
                    ["Cut Detector Same", "Manual Same"],
                    title_1,
                    mode=mode,
                )
                results.append([exp, "On the same divisions, is Cut-Detector equivalent to manual (t1-t0)?", condition, "-", delta, "TOST Wilcoxon signed-rank", p_value, title_1])

                # Same onset
                printer.print("# Same onset #")
                cd_onset = Division.get_onsets(both, div_type="cd")
                manual_onset = Division.get_onsets(both, div_type="manual")
                message, p_value = wilcoxon_equivalence(cd_onset, manual_onset, delta=delta, verbose=verbose)
                results.append([exp, "On the same divisions, is Cut-Detector equivalent to manual (t0)?", condition, "-", delta, "TOST Wilcoxon signed-rank", p_value, message])

                # Same cut
                printer.print("# Same cut #")
                cd_cut = Division.get_cut_time(both, div_type="cd")
                manual_cut = Division.get_cut_time(both, div_type="manual")
                message, p_value = wilcoxon_equivalence(cd_cut, manual_cut, delta=delta, verbose=verbose)
                results.append([exp, "On the same divisions, is Cut-Detector equivalent to manual (t1)?", condition, "-", delta, "TOST Wilcoxon signed-rank", p_value, message])

                # False positive
                printer.print("# False positive #")
                fp_cuts = Division.get_cuts(only_cd, div_type="cd")
                cd_cuts = Division.get_cuts(both, div_type="cd")
                title_2, p_value = mannwhitneyu_equivalence(cd_cuts, fp_cuts, delta=delta, verbose=verbose)
                fig_2 = create_plot(
                    [cd_cuts, fp_cuts],
                    ["Cut Detector Same", "FP"],
                    title_2,
                    mode=mode,
                )
                results.append([exp, "Are FP cuts equivalent to detected cuts (t1-t0)?", condition, "-", delta, "TOST Mann-Whitney U", p_value, title_2])

                # False negative
                printer.print("# False negative #")
                fn_cuts = Division.get_cuts(only_manual, div_type="manual")
                manual_cuts = Division.get_cuts(both, div_type="manual")
                title_3, p_value = mannwhitneyu_equivalence(
                    manual_cuts, fn_cuts, delta=delta, verbose=verbose
                )
                fig_3 = create_plot(
                    [manual_cuts, fn_cuts],
                    ["Manual Same", "FN"],
                    title_3,
                    mode=mode,
                )
                results.append([exp, "Are FN cuts equivalent to detected cuts (t1-t0)?", condition, "-", delta, "TOST Mann-Whitney U", p_value, title_3])

            printer.print("\n### Difference ###")

            # Same
            printer.print("# Same #")
            cd_cuts = Division.get_cuts(both, div_type="cd")
            manual_cuts = Division.get_cuts(both, div_type="manual")
            message, p_value = wilcoxon_difference(cd_cuts, manual_cuts, verbose=verbose)
            results.append([exp, "On the same divisions, is Cut-Detector different from manual (t1-t0)?", condition, "-", "-", "Wilcoxon signed-rank", p_value, message])

            # False negative
            printer.print("# False negative #")
            fn_cuts = Division.get_cuts(only_manual, div_type="manual")
            manual_cuts = Division.get_cuts(both, div_type="manual")
            message, p_value = mannwhitneyu_difference(manual_cuts, fn_cuts, verbose=verbose)
            results.append([exp, "Are FN cuts different from detected cuts (t1-t0)?", condition, "-", "-", "Mann-Whitney U", p_value, message])

        if show or save_fig:
                # Combine plots using make_subplots
                fig = make_subplots(
                    rows=3,
                    cols=1,
                    subplot_titles=(title_1, title_2, title_3),
                    specs=[
                        [{"type": "box"}],
                        [{"type": "box"}],
                        [{"type": "box"}],
                    ],
                )

                # Add traces to the subplots
                fig.add_trace(fig_1.data[0], row=1, col=1)
                fig.add_trace(fig_1.data[1], row=1, col=1)

                fig.add_trace(fig_2.data[0], row=2, col=1)
                fig.add_trace(fig_2.data[1], row=2, col=1)

                fig.add_trace(fig_3.data[0], row=3, col=1)
                fig.add_trace(fig_3.data[1], row=3, col=1)

                # Update layout for better appearance
                fig.update_layout(height=900, width=900, title_text=numbers_summary)

                # Show the figure
                if show:
                    fig.show()
                if save_fig:
                    assert save_folder
                    fig.write_html(
                        os.path.join(save_folder, f"condition_{condition}_{hour_date_string}.html")
                    )

    for exp in experiments_set:

        # Compare conditions vs Control
        control_divisions = Division.split_divisions(divisions, "Control", exp)
        cd_control = control_divisions[0] + control_divisions[2]
        manual_control = control_divisions[1] + control_divisions[2]
        control_cd_cuts = Division.get_cuts(cd_control, div_type="cd")
        control_manual_cuts = Division.get_cuts(manual_control, div_type="manual")
        
        for condition in conditions_vs_control:
            printer.print(f"\n### Condition {condition} vs Control ###")
            condition_divisions = Division.split_divisions(
                divisions, condition, exp
            )

            for delta in deltas:

                printer.print(f"\n### Delta {delta} ###")

                # Compare Cut Detector
                printer.print("# Cut Detector #")
                cd_condition = condition_divisions[0] + condition_divisions[2]
                condition_cuts = Division.get_cuts(cd_condition, div_type="cd")
                title_1, p_value = mannwhitneyu_equivalence(
                    control_cd_cuts, condition_cuts, delta=delta, verbose=verbose
                )
                fig_1 = create_plot(
                    [control_cd_cuts, condition_cuts],
                    ["Control", condition],
                    "",
                    mode=mode,
                )
                results.append([exp, "Is control equivalent to condition (Cut-Detector, t1-t0)?", "Control", condition, delta, "TOST Mann-Withney U", p_value, title_1])

                # Compare Manual
                printer.print("# Manual #")
                manual_condition = condition_divisions[1] + condition_divisions[2]
                condition_cuts = Division.get_cuts(
                    manual_condition, div_type="manual"
                )
                title_2, p_value = mannwhitneyu_equivalence(
                    control_manual_cuts, condition_cuts, delta=delta, verbose=verbose
                )
                fig_2 = create_plot(
                    [control_manual_cuts, condition_cuts],
                    ["Control", condition],
                    "",
                    mode=mode,
                )
                results.append([exp, "Is control equivalent to condition (manual, t1-t0)?", "Control", condition, delta, "TOST Mann-Withney U", p_value, title_2])

            printer.print("\n### Difference ###")

            # Compare Cut Detector
            printer.print("# Cut Detector #")
            cd_condition = condition_divisions[0] + condition_divisions[2]
            condition_cuts = Division.get_cuts(cd_condition, div_type="cd")
            message, p_value = mannwhitneyu_difference(control_cd_cuts, condition_cuts, verbose=verbose)
            results.append([exp, "Is control different from condition (Cut-Detector, t1-t0)?", "Control", condition, "-", "TOST Mann-Withney U", p_value, message])

            # Compare Manual
            printer.print("# Manual #")
            manual_condition = condition_divisions[1] + condition_divisions[2]
            condition_cuts = Division.get_cuts(
                manual_condition, div_type="manual"
            )
            message, p_value = mannwhitneyu_difference(
                control_manual_cuts, condition_cuts, verbose=verbose
            )
            results.append([exp, "Is control different from condition (manual, t1-t0)?", "Control", condition, "-", "TOST Mann-Withney U", p_value, message])

        if show or save_fig:

            # Combine plots using make_subplots
            fig = make_subplots(
                rows=2,
                cols=1,
                subplot_titles=(title_1, title_2),
                specs=[
                    [{"type": "box"}],
                    [{"type": "box"}],
                ],
            )

            # Add traces to the subplots
            fig.add_trace(fig_1.data[0], row=1, col=1)
            fig.add_trace(fig_1.data[1], row=1, col=1)

            fig.add_trace(fig_2.data[0], row=2, col=1)
            fig.add_trace(fig_2.data[1], row=2, col=1)

            fig.add_trace(fig_2.data[0], row=2, col=1)
            fig.add_trace(fig_2.data[1], row=2, col=1)

            # Show the figure
            if show:
                fig.show()
            if save_fig:
                assert save_folder
                fig.write_html(
                    os.path.join(save_folder, f"control_vs_condition_{condition}_{hour_date_string}.html")
                )

    # Save results as csv
    results_df = pd.DataFrame(results, columns=results_header)
    results_df.to_csv(os.path.join(save_folder, f"results_{hour_date_string}.csv"), index=False, sep=";")
