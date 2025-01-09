import os
import pandas as pd
from plotly.subplots import make_subplots

from developers.training_and_evaluation.global_evaluation.division import (
    Division,
)
from developers.training_and_evaluation.global_evaluation.statistical_tests import (
    create_plot,
    mannwhitneyu_equivalence,
    wilcoxon_equivalence,
)


def main(
    matched_csv, manual_csv, save_folder, max_cytokinesis_frame, conditions
):
    divisions = []

    # Read matched csv
    df = pd.read_csv(matched_csv, delimiter=";")
    sheet_data = [df[col].tolist() for col in df.columns]
    for idx in range(len(sheet_data[0])):
        division_data = [col[idx] for col in sheet_data]
        if len(division_data[10]) > 3:  # impossible detection
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

    # Compare Cut Detector and manual annotations
    for condition in conditions:
        only_cd, only_manual, both = Division.split_divisions(
            divisions, condition
        )
        numbers_summary = f"Condition {condition} - Same: {len(both)}, FP: {len(only_cd)}, FN: {len(only_manual)}"

        # Same
        cd_cuts = Division.get_cuts(both, div_type="cd")
        manual_cuts = Division.get_cuts(both, div_type="manual")
        title_1 = wilcoxon_equivalence(cd_cuts, manual_cuts, delta=2)
        fig_1 = create_plot(
            [cd_cuts, manual_cuts],
            ["Cut Detector Same", "Manual Same"],
            title_1,
        )

        # False positive
        fp_cuts = Division.get_cuts(only_cd, div_type="cd")
        cd_cuts = Division.get_cuts(both, div_type="cd")
        title_2 = mannwhitneyu_equivalence(cd_cuts, fp_cuts, delta=2)
        fig_2 = create_plot(
            [cd_cuts, fp_cuts], ["Cut Detector Same", "FP"], title_2
        )

        # False negative
        fn_cuts = Division.get_cuts(only_manual, div_type="manual")
        manual_cuts = Division.get_cuts(both, div_type="manual")
        title_3 = mannwhitneyu_equivalence(manual_cuts, fn_cuts, delta=2)
        fig_3 = create_plot(
            [manual_cuts, fn_cuts], ["Manual Same", "FN"], title_3
        )

        # Combine plots using make_subplots
        fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=(title_1, title_2, title_3),
            specs=[[{"type": "box"}], [{"type": "box"}], [{"type": "box"}]],
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
        fig.show()
        fig.write_html(
            os.path.join(save_folder, f"condition_{condition}.html")
        )


if __name__ == "__main__":
    MATCHED_CSV = r"C:\Users\thoma\Downloads\RE_ Bilan journée hier + nouvelle version Cut Detector\results EXP1 - CutD-ALL_matched.csv"
    MANUAL_CSV = r"C:\Users\thoma\Downloads\RE_ Bilan journée hier + nouvelle version Cut Detector\Manual annotation - Cut_D auto comparaison ALL EXP1.csv"
    SAVE_FOLDER = r"C:\Users\thoma\Downloads"

    main(
        MATCHED_CSV,
        MANUAL_CSV,
        SAVE_FOLDER,
        max_cytokinesis_frame=216,
        conditions=["Control"],
    )
