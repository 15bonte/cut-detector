import argparse

from cut_detector.widget_functions.distribution_comparison import perform_distribution_comparison

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matched_csv_file",
        default=r"C:\Users\messa\OneDrive\Thomas\OneDrive\Documents\Doctorat\Pasteur\Annotations Pasteur final\results - CutD-ALL_matched.csv",
        help="Matched CSV file",
    )
    parser.add_argument(
        "--manual_csv_file",
        default=r"C:\Users\messa\OneDrive\Thomas\OneDrive\Documents\Doctorat\Pasteur\Annotations Pasteur final\Manual annotation - Cut_D auto comparaison ALL.csv",
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

    perform_distribution_comparison(
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
