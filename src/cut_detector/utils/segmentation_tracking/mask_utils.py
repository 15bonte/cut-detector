"""From segmentation mask to list of coordinates.
Most of the code below is inspired from Trackmate (https://github.com/trackmate-sc/TrackMate)"""

import numpy as np


class MaskPolygon:
    """Polygon class to store the coordinates of a polygon."""

    def __init__(self, x, y, count):
        self.x = x
        self.y = y
        self.count = count

    def get_length(self, is_line):
        """Calculate the length of the polygon."""
        length = 0.0
        for i in range(self.count - 1):
            dx = self.x[i + 1] - self.x[i]
            dy = self.y[i + 1] - self.y[i]
            length += np.sqrt(dx * dx + dy * dy)
        if not is_line:
            dx = self.x[0] - self.x[self.count - 1]
            dy = self.y[0] - self.y[self.count - 1]
            length += np.sqrt(dx * dx + dy * dy)
        return length


class Outline:
    """This class implements a Cartesian polygon in progress. The edges are
    supposed to be parallel to the x or y axis. It is implemented as a deque
    to be able to add points to both sides."""

    def __init__(self):
        self.grow = 10
        self.reserved = self.grow
        self.first = int(self.grow / 2)
        self.last = int(self.grow / 2)

        self.x = [0] * self.reserved
        self.y = [0] * self.reserved

    def needs(self, needed_at_begin, needed_at_end):
        """Makes sure that enough free space is available at the beginning and
        end of the list, by enlarging the arrays if required"""
        if (
            needed_at_begin > self.first
            or needed_at_end > self.reserved - self.last
        ):
            extra_space = max(
                self.grow, abs(self.x[self.last - 1] - self.x[self.first])
            )
            new_size = (
                self.reserved + needed_at_begin + needed_at_end + extra_space
            )
            new_first = needed_at_begin + extra_space // 2
            new_x = [0] * new_size
            new_y = [0] * new_size
            new_x[new_first : new_first + (self.last - self.first)] = self.x[
                self.first : self.last
            ]
            new_y[new_first : new_first + (self.last - self.first)] = self.y[
                self.first : self.last
            ]
            self.x = new_x
            self.y = new_y
            self.last += new_first - self.first
            self.first = new_first
            self.reserved = new_size

    def append_outline(self, o):
        """Merge with another Outline by adding it at the end. Thereafter, the
        other outline must not be used any more."""
        size = self.last - self.first
        o_size = o.last - o.first
        if size <= o.first and o_size > self.reserved - self.last:
            # We don't have enough space in our own array but in that of 'o'
            o.x[o.first - size : o.first] = self.x[self.first : self.last]
            o.y[o.first - size : o.first] = self.y[self.first : self.last]
            self.x = o.x
            self.y = o.y
            self.first = o.first - size
            self.last = o.last
            self.reserved = o.reserved
        else:
            # Append to our own array
            self.needs(0, o_size)
            self.x[self.last : self.last + o_size] = o.x[o.first : o.last]
            self.y[self.last : self.last + o_size] = o.y[o.first : o.last]
            self.last += o_size

    def append_point(self, x, y):
        """Adds point x, y at the end of the list"""
        if self.last - self.first >= 2 and collinear(
            self.x[self.last - 2],
            self.y[self.last - 2],
            self.x[self.last - 1],
            self.y[self.last - 1],
            x,
            y,
        ):
            # Replace previous point
            self.x[self.last - 1] = x
            self.y[self.last - 1] = y
        else:
            self.needs(0, 1)  # Ensure space for new point
            self.x[self.last] = x
            self.y[self.last] = y
            self.last += 1

    def prepend_point(self, x, y):
        """Adds point x, y at the beginning of the list"""
        if self.last - self.first >= 2 and collinear(
            self.x[self.first + 1],
            self.y[self.first + 1],
            self.x[self.first],
            self.y[self.first],
            x,
            y,
        ):
            # Replace previous point
            self.x[self.first] = x
            self.y[self.first] = y
        else:
            self.needs(1, 0)  # Ensure space for new point at the beginning
            self.first -= 1
            self.x[self.first] = x
            self.y[self.first] = y

    def prepend_outline(self, o):
        """Merge with another Outline by adding it at the beginning. Thereafter,
        the other outline must not be used any more."""
        size = self.last - self.first
        o_size = o.last - o.first
        if size <= o.reserved - o.last and o_size > self.first:
            # We don't have enough space in our own array but in that of 'o'
            # Append our own data to that of 'o'
            o.x[o.last : o.last + size] = self.x[self.first : self.last]
            o.y[o.last : o.last + size] = self.y[self.first : self.last]
            self.x = o.x
            self.y = o.y
            self.first = o.first
            self.last = o.last + size
            self.reserved = o.reserved
        else:
            # Prepend to our own array
            self.needs(o_size, 0)
            self.first -= o_size
            self.x[self.first : self.first + o_size] = o.x[o.first : o.last]
            self.y[self.first : self.first + o_size] = o.y[o.first : o.last]

    def get_polygon(self):
        """Optimize out intermediate points of straight lines (created, e.g., by merging outlines)"""
        i = self.first + 1
        j = self.first + 1

        while i + 1 < self.last:
            if collinear(
                self.x[j - 1],
                self.y[j - 1],
                self.x[j],
                self.y[j],
                self.x[j + 1],
                self.y[j + 1],
            ):
                # Merge i + 1 into i
                self.last -= 1
            else:
                if i != j:
                    self.x[i] = self.x[j]
                    self.y[i] = self.y[j]
                i += 1
            j += 1

        # Wraparound
        if collinear(
            self.x[j - 1],
            self.y[j - 1],
            self.x[j],
            self.y[j],
            self.x[self.first],
            self.y[self.first],
        ):
            self.last -= 1
        else:
            self.x[i] = self.x[j]
            self.y[i] = self.y[j]

        if self.last - self.first > 2 and collinear(
            self.x[self.last - 1],
            self.y[self.last - 1],
            self.x[self.first],
            self.y[self.first],
            self.x[self.first + 1],
            self.y[self.first + 1],
        ):
            self.first += 1

        count = self.last - self.first
        x_new = self.x[self.first : self.first + count]
        y_new = self.y[self.first : self.first + count]

        return MaskPolygon(x_new, y_new, count)


def collinear(x1, y1, x2, y2, x3, y3):
    """Implement the logic to check if three points are collinear"""
    return (x2 - x1) * (y3 - y2) == (y2 - y1) * (x3 - x2)


def from_labeling_with_roi(segmentation_mask):
    nb_roi = np.max(segmentation_mask)
    polygons = []
    for roi in range(1, nb_roi + 1):
        # Define ROI bounding box
        roi_mask = segmentation_mask == roi
        roi_pixels = np.where(roi_mask)
        roi_min_y = int(np.min(roi_pixels[0]))
        roi_max_y = int(np.max(roi_pixels[0]))
        roi_min_x = int(np.min(roi_pixels[1]))
        roi_max_x = int(np.max(roi_pixels[1]))
        # Extract submask
        sub_mask = roi_mask[roi_min_y:roi_max_y, roi_min_x:roi_max_x]
        # Convert submask to polygon
        local_polygons = mask_to_polygons(sub_mask)
        # Sort them by count and keep the biggest one
        local_polygons.sort(key=lambda x: x.count, reverse=True)
        local_polygon = local_polygons[0]
        # Move polygon to global coordinates
        # NB: Trackmate switches x and y
        local_polygon.x = [lx + roi_min_y for lx in local_polygon.x]
        local_polygon.y = [ly + roi_min_x for ly in local_polygon.y]
        polygons.append(local_polygon)
    return polygons


def mask_to_polygons(mask):
    """Parse a 2D mask and return a list of polygons for the external contours of white objects.
    Warning: cannot deal with holes, they are simply ignored.
    Copied and adapted from ImageJ1 code by Wayne Rasband."""
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
                            outline[x + 1] = outline[x] = Outline()
                            outline[x].append_point(x + 1, y)
                            outline[x].append_point(x, y)
                        else:
                            outline[x] = outline[x + 1]
                            outline[x + 1] = None
                            outline[x].append_point(x, y)
                    elif outline[x + 1] is None:
                        if x == x_after_lower_right_corner:
                            outline[x + 1] = outline[x]
                            outline[x] = o_after_lower_right_corner
                            outline[x].append_point(x, y)
                            outline[x + 1].prepend_point(x + 1, y)
                        else:
                            outline[x + 1] = outline[x]
                            outline[x] = None
                            outline[x + 1].prepend_point(x + 1, y)
                    elif outline[x + 1] == outline[x]:
                        if (
                            x < w - 1
                            and y < h
                            and x != x_after_lower_right_corner
                            and not this_row[x + 2]
                            and prev_row[x + 2]
                        ):  # at lower right corner & next pxl deselected
                            outline[x] = None
                            outline[x + 1].prepend_point(x + 1, y)
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
                        outline[x].prepend_outline(outline[x + 1])
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
                    outline[x].append_point(x, y + 1)
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
                            outline[x] = outline[x + 1] = Outline()
                            outline[x].append_point(x, y)
                            outline[x].append_point(x + 1, y)
                        else:
                            outline[x] = outline[x + 1]
                            outline[x + 1] = None
                            outline[x].prepend_point(x, y)
                    elif outline[x + 1] is None:
                        if x == x_after_lower_right_corner:
                            outline[x + 1] = outline[x]
                            outline[x] = o_after_lower_right_corner
                            outline[x].prepend_point(x, y)
                            outline[x + 1].prepend_point(x + 1, y)
                        else:
                            outline[x + 1] = outline[x]
                            outline[x] = None
                            outline[x + 1].append_point(x + 1, y)
                    elif outline[x + 1] == outline[x]:
                        if (
                            x < w - 1
                            and y < h
                            and x != x_after_lower_right_corner
                            and this_row[x + 2]
                            and not prev_row[x + 2]
                        ):
                            outline[x] = None
                            outline[x + 1].append_point(x + 1, y)
                            x_after_lower_right_corner = x + 1
                            o_after_lower_right_corner = outline[x + 1]
                        else:
                            polygons.append(outline[x].get_polygon())
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
                            outline[x].append_point(x + 1, y)
                            outline[x + 1].prepend_point(x + 1, y)
                            x_after_lower_right_corner = x + 1
                            o_after_lower_right_corner = outline[x]
                            outline[x] = None
                        else:
                            outline[x].append_outline(outline[x + 1])  # merge
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
                        outline[x].prepend_point(x, y + 1)

    return polygons


def get_interpolated_polygon(p, interval):
    """Returns an interpolated version of this selection
    with points spaced abs('interval') pixels apart.
    If 'interval' is negative, the program is allowed to decrease abs('interval')
    so that the last segment will hit the end point"""

    allow_to_adjust = interval < 0
    interval = abs(interval)

    length = p.get_length(is_line=False)

    n_points = len(p.x)
    if n_points < 2:
        return p
    if abs(interval) < 0.01:
        raise ValueError("Interval must be >= 0.01")

    p.x.append(p.x[0])
    p.y.append(p.y[0])

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
        dest_x_arr[0] = p.x[0]
        dest_y_arr[0] = p.y[0]
        src_ptr = 0
        dest_ptr = 0
        x_a = p.x[0]  # start of current segment
        y_a = p.y[0]

        while src_ptr < n_points - 1:  # collect vertices
            x_c = dest_x_arr[dest_ptr]  # center circle
            y_c = dest_y_arr[dest_ptr]
            x_b = p.x[src_ptr + 1]  # end of current segment
            y_b = p.y[src_ptr + 1]
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
                x_a = p.x[src_ptr]
                y_a = p.y[src_ptr]

        dest_ptr += 1
        dest_x_arr[dest_ptr] = p.x[n_points - 1]
        dest_y_arr[dest_ptr] = p.y[n_points - 1]
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

    f_poly = MaskPolygon(x_points_new, y_points_new, len(x_points_new))
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


def perpendicular_distance(px, py, vx, vy, wx, wy):
    """Calculate the perpendicular distance from point (px, py) to line segment (vx, vy) - (wx, wy)"""
    l2 = (vx - wx) ** 2 + (vy - wy) ** 2  # length squared
    if l2 == 0:
        return np.sqrt((px - vx) ** 2 + (py - vy) ** 2)
    t = ((px - vx) * (wx - vx) + (py - vy) * (wy - vy)) / l2
    t = max(0, min(1, t))
    projection_x = vx + t * (wx - vx)
    projection_y = vy + t * (wy - vy)
    return np.sqrt((px - projection_x) ** 2 + (py - projection_y) ** 2)


def douglas_peucker_core(points_list, s, e, epsilon, result_list):
    """Recursive Douglas-Peucker line simplification."""
    # Find the point with the maximum distance
    dmax = 0
    index = 0

    start = s
    end = e - 1
    for i in range(start + 1, end):
        # Point
        px, py = points_list[i]
        # Start
        vx, vy = points_list[start]
        # End
        wx, wy = points_list[end]
        d = perpendicular_distance(px, py, vx, vy, wx, wy)
        if d > dmax:
            index = i
            dmax = d

    # If max distance is greater than epsilon, recursively simplify
    if dmax > epsilon:
        # Recursive call
        douglas_peucker_core(points_list, s, index, epsilon, result_list)
        douglas_peucker_core(points_list, index, e, epsilon, result_list)
    else:
        if end - start > 0:
            result_list.append(points_list[start])
            result_list.append(points_list[end])
        else:
            result_list.append(points_list[start])


def douglas_peucker(points_list, epsilon):
    """Given a curve composed of line segments find a similar curve with fewer
    points.
    The Ramer–Douglas–Peucker algorithm (RDP) is an algorithm for reducing
    the number of points in a curve that is approximated by a series of
    points."""
    result_list = []
    douglas_peucker_core(
        points_list, 0, len(points_list), epsilon, result_list
    )
    return result_list


def simplify(p, interval, epsilon):
    """Simplify a polygon by reducing the number of points using the Douglas-Peucker algorithm."""
    p = get_interpolated_polygon(p, interval)
    points = []
    for i in range(p.count):
        points.append([p.x[i], p.y[i]])
    simplified_points = douglas_peucker(points, epsilon)
    x_simplified, y_simplified = zip(*simplified_points)
    return MaskPolygon(
        list(x_simplified), list(y_simplified), len(x_simplified)
    )


def signed_area(x, y):
    """Compute the signed area of a polygon."""
    n = len(x)
    a = 0.0
    for i in range(n - 1):
        a += x[i] * y[i + 1] - x[i + 1] * y[i]
    return (a + x[n - 1] * y[0] - x[0] * y[n - 1]) / 2.0


def centroid(x, y):
    """Calculate the centroid of a polygon."""
    area = signed_area(x, y)
    ax = 0.0
    ay = 0.0
    n = len(x)
    for i in range(n - 1):
        w = x[i] * y[i + 1] - x[i + 1] * y[i]
        ax += (x[i] + x[i + 1]) * w
        ay += (y[i] + y[i + 1]) * w

    w0 = x[n - 1] * y[0] - x[0] * y[n - 1]
    ax += (x[n - 1] + x[0]) * w0
    ay += (y[n - 1] + y[0]) * w0

    return [ax / 6.0 / area, ay / 6.0 / area]
