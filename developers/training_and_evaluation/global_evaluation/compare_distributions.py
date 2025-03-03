import argparse
import os
import pandas as pd
from plotly.subplots import make_subplots

from developers.training_and_evaluation.global_evaluation.division import (
    Division,
)
from developers.training_and_evaluation.global_evaluation.statistical_tests import (
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


def main(
    matched_csv,
    manual_csv,
    save_folder,
    max_cytokinesis_frame,
    conditions_vs_manual,
    conditions_vs_control,
    deltas,
    mode,
    show,
    save_fig,
    verbose
):
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
        Division.generate_csv_summary(divisions, save_folder)

    results_header = ["EXP", "Comparison", "Condition 1", "Condition 2", "Delta", "Test", "p-value", "Result"]
    results = []

    experiments_set = set(sheet_data[0] + ["EXP"])  # to gather all

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
                results.append([exp, "Are FP cuts equivalent to detected cuts (t1-t0)?", condition, "-", delta, "TOST Mann-Whitney U", p_value, title_3])

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
                        os.path.join(save_folder, f"condition_{condition}.html")
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
                    os.path.join(save_folder, f"control_vs_condition_{condition}.html")
                )

    # Save results as csv
    results_df = pd.DataFrame(results, columns=results_header)
    results_df.to_csv(os.path.join(save_folder, "results.csv"), index=False, sep=";")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matched_csv_file",
        default=r"C:\Users\messa\Downloads\RE_ Bilan journée hier + nouvelle version Cut Detector\results - CutD-ALL_matched.csv",
        help="Matched CSV file",
    )
    parser.add_argument(
        "--manual_csv_file",
        default=r"C:\Users\messa\Downloads\RE_ Bilan journée hier + nouvelle version Cut Detector\Manual annotation - Cut_D auto comparaison ALL.csv",
        help="Manual CSV file",
    )
    parser.add_argument(
        "--save_folder",
        default=r"C:\Users\messa\OneDrive\Thomas\OneDrive\Documents\Doctorat\Pasteur\Résultats tests statistiques",
        help="Path to save the plots",
    )
    parser.add_argument(
        "--conditions_vs_manual",
        nargs="*",
        type=str,
        help="List of conditions to compare with manual annotations, among Control, MICAL1, CEP55, Spastin",
        default=["Control", "MICAL1", "CEP55"],
    )
    parser.add_argument(
        "--conditions_vs_control",
        nargs="*",
        type=str,
        help="List of conditions to compare to control, among MICAL1, CEP55, Spastin",
        default=["MICAL1", "CEP55"],
    )
    parser.add_argument(
        "--deltas",
        nargs="*",
        type=int,
        help="List of deltas to consider",
        default=[10, 20, 30, 40],
    )
    parser.add_argument(
        "--max_cytokinesis_frame",
        default=216,
        help="Ignore divisions after this frame",
    )
    parser.add_argument(
        "--verbose",
        default=False,
        help="Verbose mode",
    )
    args = parser.parse_args()

    main(
        args.matched_csv_file,
        args.manual_csv_file,
        args.save_folder,
        max_cytokinesis_frame=args.max_cytokinesis_frame,
        conditions_vs_manual=args.conditions_vs_manual,
        conditions_vs_control=args.conditions_vs_control,
        deltas=args.deltas,
        mode="cumulative",  # cumulative or box
        show=False,
        save_fig=False,
        verbose=args.verbose
    )
