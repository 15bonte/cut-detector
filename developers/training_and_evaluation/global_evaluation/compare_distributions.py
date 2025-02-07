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
):
    # Remove empty strings from conditions_vs_manual and conditions_vs_control
    conditions_vs_manual = [condition for condition in conditions_vs_manual if condition]
    conditions_vs_control = [condition for condition in conditions_vs_control if condition]

    divisions = []

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

    Division.generate_csv_summary(divisions, save_folder)

    # Compare Cut Detector and manual annotations
    for condition in conditions_vs_manual:

        print(f"\n ### Condition {condition} - Cut Detector vs Manual ###")
        only_cd, only_manual, both = Division.split_divisions(
            divisions, condition
        )
        numbers_summary = f"\n Condition {condition} - Same: {len(both)}, FP: {len(only_cd)}, FN: {len(only_manual)} \n"
        print(numbers_summary)

        for delta in deltas:

            print(f"\n ### Delta {delta} ###")

            # Same
            print("# Same #")
            cd_cuts = Division.get_cuts(both, div_type="cd")
            manual_cuts = Division.get_cuts(both, div_type="manual")
            title_1 = wilcoxon_equivalence(cd_cuts, manual_cuts, delta=delta)
            wilcoxon_difference(cd_cuts, manual_cuts)
            fig_1 = create_plot(
                [cd_cuts, manual_cuts],
                ["Cut Detector Same", "Manual Same"],
                title_1,
                mode=mode,
            )

            # Same onset
            print("# Same onset #")
            cd_onset = Division.get_onsets(both, div_type="cd")
            manual_onset = Division.get_onsets(both, div_type="manual")
            wilcoxon_equivalence(cd_onset, manual_onset, delta=delta)

            # Same cut
            print("# Same cut #")
            cd_cut = Division.get_cut_time(both, div_type="cd")
            manual_cut = Division.get_cut_time(both, div_type="manual")
            wilcoxon_equivalence(cd_cut, manual_cut, delta=delta)

            # False positive
            print("# False positive #")
            fp_cuts = Division.get_cuts(only_cd, div_type="cd")
            cd_cuts = Division.get_cuts(both, div_type="cd")
            title_2 = mannwhitneyu_equivalence(cd_cuts, fp_cuts, delta=delta)
            fig_2 = create_plot(
                [cd_cuts, fp_cuts],
                ["Cut Detector Same", "FP"],
                title_2,
                mode=mode,
            )

            # False negative
            print("# False negative #")
            fn_cuts = Division.get_cuts(only_manual, div_type="manual")
            manual_cuts = Division.get_cuts(both, div_type="manual")
            title_3 = mannwhitneyu_equivalence(
                manual_cuts, fn_cuts, delta=delta
            )
            mannwhitneyu_difference(manual_cuts, fn_cuts)
            fig_3 = create_plot(
                [manual_cuts, fn_cuts],
                ["Manual Same", "FN"],
                title_3,
                mode=mode,
            )

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
        if save_folder:
            fig.write_html(
                os.path.join(save_folder, f"condition_{condition}.html")
            )

    # Compare conditions vs Control
    control_divisions = Division.split_divisions(divisions, "Control")
    cd_control = control_divisions[0] + control_divisions[2]
    manual_control = control_divisions[1] + control_divisions[2]
    control_cd_cuts = Division.get_cuts(cd_control, div_type="cd")
    control_manual_cuts = Division.get_cuts(manual_control, div_type="manual")
    for condition in conditions_vs_control:
        print(f"\n ### Condition {condition} vs Control ###")

        for delta in deltas:

            print(f"\n ### Delta {delta} ###")
            condition_divisions = Division.split_divisions(
                divisions, condition
            )

            # Compare Cut Detector
            print("# Cut Detector #")
            cd_condition = condition_divisions[0] + condition_divisions[2]
            condition_cuts = Division.get_cuts(cd_condition, div_type="cd")
            mannwhitneyu_equivalence(
                control_cd_cuts, condition_cuts, delta=delta
            )
            title_1 = mannwhitneyu_difference(control_cd_cuts, condition_cuts)

            fig_1 = create_plot(
                [control_cd_cuts, condition_cuts],
                ["Control", condition],
                "",
                mode=mode,
            )

            # Compare Manual
            print("# Manual #")
            manual_condition = condition_divisions[1] + condition_divisions[2]
            condition_cuts = Division.get_cuts(
                manual_condition, div_type="manual"
            )
            mannwhitneyu_equivalence(
                control_manual_cuts, condition_cuts, delta=delta
            )
            title_2 = mannwhitneyu_difference(
                control_manual_cuts, condition_cuts
            )

            fig_2 = create_plot(
                [control_manual_cuts, condition_cuts],
                ["Control", condition],
                "",
                mode=mode,
            )

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
        if save_folder:
            fig.write_html(
                os.path.join(save_folder, f"control_vs_condition_{condition}.html")
            )


if __name__ == "__main__":
    EXP = "" # "EXP1", "EXP2", "EXP3" or "" (for all)
    CONDITION = "" # "Spastin" or ""

    MATCHED_CSV = r"C:\Users\messa\Downloads\RE_ Bilan journée hier + nouvelle version Cut Detector\results " + EXP + "- CutD-ALL_matched" + CONDITION + ".csv"
    MANUAL_CSV = r"C:\Users\messa\Downloads\RE_ Bilan journée hier + nouvelle version Cut Detector\Manual annotation - Cut_D auto comparaison ALL" + EXP + CONDITION + ".csv"
    SAVE_FOLDER = r"C:\Users\messa\Downloads"

    main(
        MATCHED_CSV,
        MANUAL_CSV,
        SAVE_FOLDER,
        max_cytokinesis_frame=216,
        conditions_vs_manual=["Control"], # Control, MICAL1, CEP55, Spastin
        conditions_vs_control=[""],
        deltas=[10, 20, 30, 40],
        mode="cumulative",  # cumulative or box
        show=False,
    )
