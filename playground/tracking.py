import os
import pickle
from typing import Optional
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

from cut_detector.data.tools import get_data_path
from cut_detector.utils.trackmate_track import TrackMateTrack
from cut_detector.utils.trackmate_spot import TrackMateSpot


Outline = namedtuple("Outline", ["points"])


def mask_to_polygons(mask):
    w = mask.shape[0]
    h = mask.shape[1]

    polygons = []
    prev_row = [False] * (w + 2)
    this_row = [False] * (w + 2)
    outline = [None] * (w + 1)

    for y in range(h + 1):

        b = prev_row
        prev_row = this_row
        this_row = b

        x_after_lower_right_corner = -1
        o_after_lower_right_corner = None

        this_row[1] = mask[0, y] if y < h else False

        for x in range(w + 1):

            # we need to read one pixel ahead
            if y < h and x < w - 1:
                this_row[x + 2] = mask[x + 1, y]
            elif x < w - 1:
                this_row[x + 2] = False

            if this_row[x + 1]:  # pixel (x,y) is selected
                if not prev_row[x + 1]:
                    # Upper edge of selected area:
                    # - left and right outlines are null: new outline
                    # - left null: append (line to left)
                    # - right null: prepend (line to right), or
                    # prepend&append (after lower right corner, two borders
                    # from one corner)
                    # - left == right: close (end of hole above) unless we
                    # can continue at the right
                    # - left != right: merge (prepend) unless we can
                    # continue at the right
                    if outline[x] is None:
                        if outline[x + 1] is None:
                            outline[x] = Outline([(x + 1, y), (x, y)])
                            outline[x + 1] = outline[x]
                        else:
                            outline[x] = outline[x + 1]
                            outline[x + 1] = None
                            outline[x].points.append((x, y))
                    elif outline[x + 1] is None:
                        if x == x_after_lower_right_corner:
                            outline[x + 1] = outline[x]
                            outline[x] = o_after_lower_right_corner
                            outline[x].points.append((x, y))
                            outline[x + 1].points.insert(0, (x + 1, y))
                        else:
                            outline[x + 1] = outline[x]
                            outline[x] = None
                            outline[x + 1].points.insert(0, (x + 1, y))
                    elif outline[x + 1] == outline[x]:
                        if (
                            x < w - 1
                            and y < h
                            and x != x_after_lower_right_corner
                            and not this_row[x + 2]
                            and prev_row[x + 2]
                        ):  # at lower right corner & next pxl deselected
                            outline[x] = None
                            outline[x + 1].points.insert(0, (x + 1, y))
                            x_after_lower_right_corner = x + 1
                            o_after_lower_right_corner = outline[x + 1]
                        else:  # we cannot handle holes
                            outline[x + 1] = None
                            outline[x] = (
                                o_after_lower_right_corner
                                if x == x_after_lower_right_corner
                                else None
                            )
                    else:
                        outline[x].points.extend(outline[x + 1].points)
                        for x1 in range(w + 1):
                            if x1 != x + 1 and outline[x1] == outline[x + 1]:
                                outline[x1] = outline[x]
                                outline[x + 1] = None
                                outline[x] = (
                                    o_after_lower_right_corner
                                    if x == x_after_lower_right_corner
                                    else None
                                )
                                break
                        if outline[x + 1] is not None:
                            raise RuntimeError("Assertion failed")
                if not this_row[x]:
                    if outline[x] is None:
                        raise RuntimeError("Assertion failed")
                    outline[x].points.append((x, y + 1))
            else:  # !thisRow[x + 1], i.e., pixel (x,y) is deselected
                if prev_row[x + 1]:
                    # Lower edge of selected area:
                    # - left and right outlines are null: new outline
                    # - left == null: prepend
                    # - right == null: append, or append&prepend (after
                    # lower right corner, two borders from one corner)
                    # - right == left: close unless we can continue at the
                    # right
                    # - right != left: merge (append) unless we can
                    # continue at the right
                    if outline[x] is None:
                        if outline[x + 1] is None:
                            outline[x] = outline[x + 1] = Outline(
                                [(x, y), (x + 1, y)]
                            )
                        else:
                            outline[x] = outline[x + 1]
                            outline[x + 1] = None
                            outline[x].points.insert(0, (x, y))
                    elif outline[x + 1] is None:
                        if x == x_after_lower_right_corner:
                            outline[x + 1] = outline[x]
                            outline[x] = o_after_lower_right_corner
                            outline[x].points.insert(0, (x, y))
                            outline[x + 1].points.append((x + 1, y))
                        else:
                            outline[x + 1] = outline[x]
                            outline[x] = None
                            outline[x + 1].points.append((x + 1, y))
                    elif outline[x + 1] == outline[x]:
                        if (
                            x < w - 1
                            and y < h
                            and x != x_after_lower_right_corner
                            and this_row[x + 2]
                            and not prev_row[x + 2]
                        ):
                            outline[x] = None
                            outline[x + 1].points.insert(
                                0, (x + 1, y)
                            )  # polygons.add( outline[ x ].getPolygon() );
                            x_after_lower_right_corner = x + 1
                            o_after_lower_right_corner = outline[x + 1]
                        else:
                            polygons.append(outline[x].points)
                            outline[x + 1] = None
                            outline[x] = (
                                o_after_lower_right_corner
                                if x == x_after_lower_right_corner
                                else None
                            )
                    else:
                        if (
                            x < w - 1
                            and y < h
                            and x != x_after_lower_right_corner
                            and this_row[x + 2]
                            and not prev_row[x + 2]
                        ):
                            outline[x].points.append((x + 1, y))
                            outline[x + 1].points.insert(0, (x + 1, y))
                            x_after_lower_right_corner = x + 1
                            o_after_lower_right_corner = outline[x]
                            outline[x] = None
                        else:
                            outline[x].points.extend(outline[x + 1].points)
                            for x1 in range(w + 1):
                                if (
                                    x1 != x + 1
                                    and outline[x1] == outline[x + 1]
                                ):
                                    outline[x1] = outline[x]
                                    outline[x + 1] = None
                                    outline[x] = (
                                        o_after_lower_right_corner
                                        if x == x_after_lower_right_corner
                                        else None
                                    )
                                    break
                            if outline[x + 1] is not None:
                                raise RuntimeError("Assertion failed")
                    if this_row[x]:
                        if outline[x] is None:
                            raise RuntimeError("Assertion failed")
                        outline[x].points.insert(0, (x, y + 1))

    return polygons


def get_length(x, y):
    dx = x[0] - x[-1]
    dy = y[0] - y[-1]
    return np.sqrt(dx * dx + dy * dy)


def get_interpolated_polygon(p, interval):
    allow_to_adjust = interval < 0
    interval = abs(interval)

    x_points = [x for x, _ in p]
    y_points = [y for _, y in p]

    length = get_length(x_points, y_points)

    n_points = len(p)
    if n_points < 2:
        return p
    if abs(interval) < 0.01:
        raise ValueError("Interval must be >= 0.01")

    x_points.append(x_points[0])
    y_points.append(y_points[0])

    n_points2 = int(10 + (length * 1.5) / interval)  # allow some headroom
    try_interval = interval
    min_diff = 1e9
    best_interval = 0
    src_ptr = 0  # index of source polygon
    dest_ptr = 0  # index of destination polygon
    dest_x_arr = np.zeros(n_points2)
    dest_y_arr = np.zeros(n_points2)
    n_trials = 50
    trial = 0

    while trial <= n_trials:
        dest_x_arr[0] = x_points[0]
        dest_y_arr[0] = y_points[0]
        src_ptr = 0
        dest_ptr = 0
        x_a = x_points[0]  # start of current segment
        y_a = y_points[0]

        while src_ptr < n_points - 1:  # collect vertices
            x_c = dest_x_arr[dest_ptr]  # center circle
            y_c = dest_y_arr[dest_ptr]
            x_b = x_points[src_ptr + 1]  # end of current segment
            y_b = y_points[src_ptr + 1]
            intersections = line_circle_intersection(
                x_a, y_a, x_b, y_b, x_c, y_c, try_interval, True
            )
            if len(intersections) >= 2:
                x_a, y_a = intersections[
                    :2
                ]  # only use first of two intersections
                dest_ptr += 1
                dest_x_arr[dest_ptr] = x_a
                dest_y_arr[dest_ptr] = y_a
            else:
                src_ptr += 1  # no intersection found, pick next segment
                x_a = x_points[src_ptr]
                y_a = y_points[src_ptr]

        dest_ptr += 1
        dest_x_arr[dest_ptr] = x_points[n_points - 1]
        dest_y_arr[dest_ptr] = y_points[n_points - 1]
        dest_ptr += 1
        if not allow_to_adjust:
            break

        n_segments = dest_ptr - 1
        dx = dest_x_arr[dest_ptr - 2] - dest_x_arr[dest_ptr - 1]
        dy = dest_y_arr[dest_ptr - 2] - dest_y_arr[dest_ptr - 1]
        last_seg = np.sqrt(dx * dx + dy * dy)

        diff = last_seg - try_interval
        if abs(diff) < min_diff:
            min_diff = abs(diff)
            best_interval = try_interval

        feedback_factor = (
            0.66  # factor <1: applying soft successive approximation
        )
        try_interval += feedback_factor * diff / n_segments
        # stop if tryInterval < 80% of interval, OR if last segment differs < 0.05 pixels
        if (
            try_interval < 0.8 * interval
            or abs(diff) < 0.05
            or trial == n_trials - 1
        ) and trial < n_trials:
            trial = n_trials
            try_interval = best_interval
        else:
            trial += 1

    # remove closing point from end of array
    dest_ptr -= 1

    x_points_new = np.zeros(dest_ptr)
    y_points_new = np.zeros(dest_ptr)
    for jj in range(dest_ptr):
        x_points_new[jj] = dest_x_arr[jj]
        y_points_new[jj] = dest_y_arr[jj]

    f_poly = [(x_points_new[i], y_points_new[i]) for i in range(dest_ptr)]
    return f_poly


def line_circle_intersection(ax, ay, bx, by, cx, cy, rad, ignore_outside):
    """
    Calculates intersections of a line segment with a circle.

    Parameters:
    ax, ay, bx, by: Points A and B of the line segment.
    cx, cy, rad: Circle center and radius.
    ignore_outside: If true, ignores intersections outside the line segment A-B.

    Returns:
    A list of 0, 2, or 4 coordinates (for 0, 1, or 2 intersection points).
    If two intersection points are returned, they are listed in travel direction A->B.
    """

    # Calculate differences and lengths
    dx_ac = cx - ax
    dy_ac = cy - ay
    len_ac = np.sqrt(dx_ac * dx_ac + dy_ac * dy_ac)

    dx_ab = bx - ax
    dy_ab = by - ay

    # Calculate B2 and C2
    x_b2 = np.sqrt(dx_ab * dx_ab + dy_ab * dy_ab)

    phi1 = np.arctan2(dy_ab, dx_ab)  # amount of rotation
    phi2 = np.arctan2(dy_ac, dx_ac)
    phi3 = phi1 - phi2
    x_c2 = len_ac * np.cos(phi3)
    y_c2 = len_ac * np.sin(phi3)  # rotation & translation is done

    if abs(y_c2) > rad:
        return []  # no intersection found

    half_chord = np.sqrt(rad * rad - y_c2 * y_c2)
    sect_one = x_c2 - half_chord  # first intersection point, still on x axis
    sect_two = x_c2 + half_chord  # second intersection point, still on x axis

    xy_coords = []

    if (0 <= sect_one <= x_b2) or not ignore_outside:
        sect_one_x = (
            np.cos(phi1) * sect_one + ax
        )  # undo rotation and translation
        sect_one_y = np.sin(phi1) * sect_one + ay
        xy_coords.extend([sect_one_x, sect_one_y])

    if (0 <= sect_two <= x_b2) or not ignore_outside:
        sect_two_x = (
            np.cos(phi1) * sect_two + ax
        )  # undo rotation and translation
        sect_two_y = np.sin(phi1) * sect_two + ay
        xy_coords.extend([sect_two_x, sect_two_y])

    if (
        half_chord == 0 and len(xy_coords) > 2
    ):  # tangent line returns only one intersection
        xy_coords = xy_coords[:2]

    return xy_coords


def load_tracks_and_spots(
    trackmate_tracks_path: str, spots_path: str
) -> tuple[list[TrackMateTrack], list[TrackMateSpot]]:
    """
    Load saved spots and tracks generated from Trackmate xml file.
    """
    trackmate_tracks: list[TrackMateTrack] = []
    for track_file in os.listdir(trackmate_tracks_path):
        with open(os.path.join(trackmate_tracks_path, track_file), "rb") as f:
            trackmate_track: TrackMateTrack = pickle.load(f)
            trackmate_track.adapt_deprecated_attributes()
            trackmate_tracks.append(trackmate_track)

    spots: list[TrackMateSpot] = []
    for spot_file in os.listdir(spots_path):
        with open(os.path.join(spots_path, spot_file), "rb") as f:
            spots.append(pickle.load(f))

    return trackmate_tracks, spots


def main(
    segmentation_results_path: Optional[str] = os.path.join(
        get_data_path("segmentation_results"), "example_video.bin"
    ),
    trackmate_tracks_path: Optional[str] = os.path.join(
        get_data_path("tracks"), "example_video"
    ),
    spots_path: Optional[str] = os.path.join(
        get_data_path("spots"), "example_video"
    ),
):
    # Load Cellpose results
    with open(segmentation_results_path, "rb") as f:
        cellpose_results = pickle.load(f)

    # TODO: create spots from Cellpose results
    # TODO: perform tracking using laptrack

    polygons = mask_to_polygons(cellpose_results[0])  # frame 0

    # for polygon in polygons:
    #     f_polygon = get_interpolated_polygon(polygon, interval=2)

    # Plot polygons
    for polygon in polygons:
        polygon = np.array(polygon)
        hull = ConvexHull(polygon)
        convex_hull_indices = polygon[hull.vertices][:, ::-1]  # (x, y)
        x, y = zip(*convex_hull_indices)
        plt.plot(x, y)
    plt.imshow(cellpose_results[0], cmap="gray")
    plt.show()

    # Load TrackMate results to compare... make sure they match!
    trackmate_tracks, trackmate_spots = load_tracks_and_spots(
        trackmate_tracks_path, spots_path
    )


if __name__ == "__main__":
    main()
