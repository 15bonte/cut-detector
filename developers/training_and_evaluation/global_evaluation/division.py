import csv
import os


class Division:
    def __init__(self, csv_line):

        self.exp = csv_line[0]
        self.video = csv_line[1]

        if len(csv_line) == 22:  # Cut Detector

            self.cd_metaphase = csv_line[9]
            self.cd_cytokinesis = csv_line[10]
            self.cd_first = int(csv_line[11])

            self.cd_position_x = csv_line[13]
            self.cd_position_y = csv_line[14]

            self.cytokinesis = csv_line[18]
            self.first = csv_line[19]

            self.position_x = csv_line[20]
            self.position_y = csv_line[21]

        elif len(csv_line) == 7:  # Manual

            self.cd_metaphase = -1
            self.cd_cytokinesis = -1
            self.cd_first = -1

            self.cd_position_x = -1
            self.cd_position_y = -1

            self.cytokinesis = csv_line[3]
            self.first = csv_line[4]

            self.position_x = csv_line[5]
            self.position_y = csv_line[6]

        else:
            raise ValueError("Invalid csv_line length")

    def exists_in(self, divisions):
        """Check if annotated division already exists in annotated divisions list"""
        for division in divisions:
            if (
                division.exp == self.exp
                and division.video == self.video
                and division.cytokinesis == self.cytokinesis
                and division.first == self.first
                and division.position_x == self.position_x
                and division.position_y == self.position_y
            ):
                return True
        return False

    @staticmethod
    def split_divisions(divisions, condition, exp):
        condition_divisions = [
            division for division in divisions if condition in division.video and exp in division.exp
        ]

        only_cd, only_manual, both = [], [], []
        for division in condition_divisions:
            if division.cd_metaphase == -1:
                only_manual.append(division)
            elif division.cytokinesis == -1:
                only_cd.append(division)
            else:
                both.append(division)

        return only_cd, only_manual, both

    @staticmethod
    def get_cuts(divisions, div_type, time_resolution=10):
        cuts = []
        assert div_type in ["cd", "manual"]

        for division in divisions:
            if div_type == "cd":
                cut = (
                    division.cd_first - division.cd_cytokinesis
                ) * time_resolution
            else:
                cut = (division.first - division.cytokinesis) * time_resolution
            cuts.append(cut)

        return cuts
    
    @staticmethod
    def get_onsets(divisions, div_type, time_resolution=10):
        onsets = []
        assert div_type in ["cd", "manual"]

        for division in divisions:
            if div_type == "cd":
                onset =  division.cd_cytokinesis * time_resolution
            else:
                onset = division.cytokinesis * time_resolution
            onsets.append(onset)

        return onsets
    
    @staticmethod
    def get_cut_time(divisions, div_type, time_resolution=10):
        frames = []
        assert div_type in ["cd", "manual"]

        for division in divisions:
            if div_type == "cd":
                frame = division.cd_first * time_resolution
            else:
                frame = division.first * time_resolution
            frames.append(frame)

        return frames

    def is_both(self):
        return self.cd_metaphase != -1 and self.cytokinesis != -1

    @staticmethod
    def generate_csv_summary(divisions, save_folder, hour_date_string, time_resolution=10):
        csv_lines = [
            [
                "Experiment",
                "Video",
                "CD Cytokinesis frame",
                "CD First cut frame",
                "Real Cytokinesis frame",
                "Real First cut frame",
                "CD Cut time",
                "Real Cut time",
                "Cytokinesis difference",
                "First cut difference",
                "Cut time difference",
            ]
        ]

        for division in divisions:
            csv_line = [
                division.exp,
                division.video,
                division.cd_cytokinesis,
                division.cd_first,
                division.cytokinesis,
                division.first,
            ]
            if division.is_both():
                cd_cut_time = (
                    division.cd_first - division.cd_cytokinesis
                ) * time_resolution
                real_cut_time = (
                    division.first - division.cytokinesis
                ) * time_resolution
                csv_line.extend(
                    [
                        cd_cut_time,
                        real_cut_time,
                        (division.cd_cytokinesis - division.cytokinesis)
                        * time_resolution,
                        (division.cd_first - division.first) * time_resolution,
                        cd_cut_time - real_cut_time,
                    ]
                )
            else:
                csv_line.extend(["-", "-", "-", "-"])
            csv_lines.append(csv_line)

        csv_path = os.path.join(save_folder, f"divisions_summary_{hour_date_string}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f, delimiter=";")
            for row in csv_lines:
                writer.writerow(row)
