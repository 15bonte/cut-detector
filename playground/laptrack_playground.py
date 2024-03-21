""" File to test some laptrack features
"""

""" LapTrack Tests Results

######################################################################
SUMMARY:
See https://github.com/yfukai/laptrack/issues/52
Basically LapTrack does not allow frames with no points.
So it's either removing frames with no points (losing temporal infos)
Or having frames with a single (None, None) point. This method creates a track
for every (None, None) point, not ideal but keeps temporal info.


LapTrack does not like DF where some frames are missing.

To avoid this:
- either set frame_validation to False:
    ??

- or add a line with a point at coordinates (None, None):
    ISSUE: it creates a new track_id for each (None,None)

Both method seam to work without disturbing the tracking process, though
further testing is required to be sure.


Why is that happening ?
Issue comes from 'convert_dataframe_to_coords'

Official way to avoid this ?
Not in the doc examples
Not in the Notebooks


>>>>> What about using predict instead of predict_dataframe ? <<<<<
Does not work, because the elements in coords must be 2D
So no empty list

######################################################################

How to add non-distance values (like colors):
Probably a custom distance metric

In GAP_CONTINUOUS_DATA:
Why point (27,27) is not linked with (29,29) and the rest ?
The way the system works is that:

1) it computes the potential frame bridge candidates, from present to future segment.
So basically every past segment gets a negative value, and other gets a positive value
that is the different between future start and current end. Then if that diff
is less than what is allowed for gaps, it gets added to a list. **This list
also keeps track of the squared spatial distance between the 2 points**

[at some point, maybe in 1 or 2, if a distance is greater than max_distance,
the candidate is removed]

2) Then, it lists all the available candidates spatial distances and takes,
by default, the 90% percentile of this distance distribution.
This is used as another threshold: spatial distance between points separated by gaps
greater than threshold are not connected, although spatial dist < max AND 
frame diff < max frame diff

Consequence: on the following example:
when taking 8 as frame AND p90:
4 dist values: 0 1 8 8: p90 is then around 8, so threshold linking distance is 8
(27,27) is linked to (29,29)
Taking 7 as frame gap AND p90, we only have 3 cross-gap candidates:
0 1 8. p90 is before 8, so max distance is much lower.
so much actually that (27,27) cannot be linked to (29,29)

**Here the unexpected behavior is caused by a combination of p90 AND a small and
unbalanced distance dataset**
On small datasets, changing p90 to p100 allows one to use gap=3 (as expected).
On bigger datasets, maybe this problem would not have appeared (case not tested).
So 1st try with p90 before, if required, changing to p100.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from laptrack import LapTrack
from laptrack.data_conversion import convert_dataframe_to_coords, convert_dataframe_to_coords_frame_index

# GAPLESS = All points can always be seen
# CONTINUOUS = At least 1 point is seen each frame
GAPLESS_CONTINUOUS_DATA = {
    0: [(0, 0), (29,29)],
    1: [(1, 1), (30,30)],
    2: [(2,1), (30,29)],
    3: [(1,1), (29,28)],
    4: [(0,0), (29,30)],
    5: [(1,0), (28,29)],
    6: [(2,0), (26,29)],
    7: [(3,1), (26,26)],
    8: [(2,2), (28,28)],
    9: [(1,1), (29,29)]
}

# GAP = Sometimes, a point can disappear
# CONTINUOUS = there is 1 line per frame
GAP_CONTINUOUS_DATA = {
    0: [(0, 0), (29,29)],
    1: [(1, 1), (30,30)],
    2: [(2,1)],
    3: [(1,1)],
    4: [(0,0), (29,30)],
    5: [(30,30)],
    6: [(0,0), (29,39)],
    7: [(1,1), (29,29)],
    8: [(2,2)],
    9: [(27,27)]
}

# GAP = Sometimes, some point can disappear
# MISSING = when no data is available no frame line is added to the df
GAP_MISSING_DATA = {
    0: [(0, 0), (29,29)],
    1: [(1, 1), (30,30)],
    2: [(2,1)],
    3: [(1,1)],
    4: [(0,0), (29,30)],
    5: [(30,30)],
    6: [],
    7: [],
    8: [(0,0), (30,30)],
    9: [(1,1), (29,29)]
}

GAP_MISSING_QUICKFIX_DATA = {
    0: [(0, 0), (29,29)],
    1: [(1, 1), (30,30)],
    2: [(2,1)],
    3: [(1,1)],
    4: [(0,0), (29,30)],
    5: [(30,30)],
    6: [(0,0), (30,30)],
    7: [(1,1), (29,29)]
}

# GAP = Sometimes, some point can disappear
# NONE = when no data is available a frame with (None, None) is added
GAP_NONE_DATA = {
    0: [(0, 0), (29,29)],
    1: [(1, 1), (30,30)],
    2: [(2,1)],
    3: [(1,1)],
    4: [(0,0), (29,30)],
    5: [(30,30)],
    6: [(None, None)],
    7: [(None, None)],
    8: [(0,0), (30,30)],
    9: [(1,1), (29,29)]
}

FAKE_DATA = GAP_CONTINUOUS_DATA


def main():
    laptrack_run()
    # investigate_convert_dataframe_to_coords_frame_index()
    # deep_investigate_convert_dataframe_to_coords_frame_index(GAP_NONE_DATA)
    # deep_investigate_convert_dataframe_to_coords_frame_index(GAP_MISSING_DATA, False)
    # laptrack_run_nodf(GAPLESS_CONTINUOUS_DATA)
    # laptrack_run_nodf(GAP_MISSING_DATA)

def make_df_from_fakedata(fake_data):
    df = pd.DataFrame({
        "frame": [],
        "x": [],
        "y": []
    })
    for frame, data in fake_data.items():
        for coord in data:
            df.loc[len(df.index)] = [frame, coord[0], coord[1]]  
    print("fake_data frame:", df, sep="\n")

    return df 


def investigate_convert_dataframe_to_coords_frame_index():
    df = make_df_from_fakedata(GAP_NONE_DATA)
    conv_df = convert_dataframe_to_coords_frame_index(df, ["x", "y"], validate_frame=True)
    print("converted df:", conv_df)

    df = make_df_from_fakedata(GAP_MISSING_DATA)
    conv_df = convert_dataframe_to_coords_frame_index(df, ["x", "y"], validate_frame=False)
    print("converted df:", conv_df)


def deep_investigate_convert_dataframe_to_coords_frame_index(data, validate = True):
    df = make_df_from_fakedata(data)
    coordinate_cols = ["x", "y"]
    frame_col = "frame"
    validate_frame = True

    assert "iloc__" not in df.columns
    # df = df.copy()
    df["iloc__"] = np.arange(len(df), dtype=int)

    print("df about to be converted:", df, sep="\n")

    coords = convert_dataframe_to_coords(
        df, list(coordinate_cols) + ["iloc__"], frame_col, validate_frame=validate
    )
    print("coords:", coords, sep="\n")


def laptrack_run():
    df = make_df_from_fakedata(FAKE_DATA)
    
    max_distance = 15
    lt = LapTrack(
        track_dist_metric="sqeuclidean",  # The similarity metric for particles. See `scipy.spatial.distance.cdist` for allowed values.
        # the square of the cutoff distance for the "sqeuclidean" metric
        track_cost_cutoff=max_distance**2,
        splitting_cost_cutoff=False,  # or False for non-splitting case
        merging_cost_cutoff=False,  # or False for non-merging case
        gap_closing_max_frame_count=3, # 2 missing frames, so delta=3
        alternative_cost_percentile=100,
    )
    track_df, split_df, merge_df = lt.predict_dataframe(
        df,
        coordinate_cols=[
            "x",
            "y",
        ],  # the column names for the coordinates
        frame_col="frame",  # the column name for the frame (default "frame")
        only_coordinate_cols=False,  # if False, returned track_df includes columns not in coordinate_cols.
        # False will be the default in the major release.
        validate_frame=True,
    )

    print("laptrack df:", track_df, sep="\n")

    def visualize():
        plt.figure(figsize=(3, 3))
        frames = track_df.index.get_level_values("frame")
        frame_range = [frames.min(), frames.max()]
        k1, k2 = "x", "y"
        keys = [k1, k2]

        def get_track_end(track_id, first=True):
            df = track_df[track_df["track_id"] == track_id].sort_index(level="frame")
            return df.iloc[0 if first else -1][keys]


        for track_id, grp in track_df.groupby("track_id"):
            df = grp.reset_index().sort_values("frame")
            plt.scatter(df[k1], df[k2], c=df["frame"], vmin=frame_range[0], vmax=frame_range[1])
            for i in range(len(df) - 1):
                pos1 = df.iloc[i][keys]
                pos2 = df.iloc[i + 1][keys]
                plt.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], "-k")
            for _, row in list(split_df.iterrows()) + list(merge_df.iterrows()):
                pos1 = get_track_end(row["parent_track_id"], first=False)
                pos2 = get_track_end(row["child_track_id"], first=True)
                plt.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], "-k")

        plt.xticks([])
        plt.yticks([])

        plt.show()
    
    visualize()

def make_coords_from_fakedata(fakedata):
    frame_list = []
    for frame, data in fakedata.items():
        frame_list.append(np.array(data))
    return frame_list
    

def laptrack_run_nodf(data):
    coords = make_coords_from_fakedata(data)
    print("coords from fakedata:", coords, sep="\n")

    max_distance = 15
    lt = LapTrack(
        track_dist_metric="sqeuclidean",  # The similarity metric for particles. See `scipy.spatial.distance.cdist` for allowed values.
        # the square of the cutoff distance for the "sqeuclidean" metric
        track_cost_cutoff=max_distance**2,
        splitting_cost_cutoff=False,  # or False for non-splitting case
        merging_cost_cutoff=False,  # or False for non-merging case
        gap_closing_max_frame_count=4,
    )

    track_tree = lt.predict(
        coords=coords
    )

    print("--Track Tree--")
    for edge in list(track_tree.edges()):
        print(edge)

def spatial_laptrack_run():
    pass

if __name__ == "__main__":
    main()