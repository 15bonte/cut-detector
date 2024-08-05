from skimage.measure import find_contours
import numpy as np

from ..cell_spot import CellSpot


def get_spots_from_frame(
    frame: int, cellpose_result: np.ndarray
) -> tuple[int, list[CellSpot]]:
    """Extract spots from a single frame.

    Parameters
    ----------
    frame : int
        Frame number.
    cellpose_result : np.ndarray
        TYX

    Returns
    -------
    list[CellSpot]
        List of cell spots.
    """
    # Pad cellpose results to ensure that the contours are closed
    max_cellpose = np.max(cellpose_result)
    padded_cellpose_result = np.pad(
        cellpose_result, pad_width=1, mode="constant"
    )

    cell_spots = []
    for i in range(1, max_cellpose + 1):
        contours = find_contours(padded_cellpose_result == i)
        sorted_contours = sorted(
            contours,
            key=lambda contour: contour.shape[0],
            reverse=True,
        )
        # -1 to remove padding
        list_y = [int(l[0] - 1) for l in sorted_contours[0]]
        list_x = [int(l[1] - 1) for l in sorted_contours[0]]
        abs_min_x, abs_max_x, abs_min_y, abs_max_y = (
            np.abs(np.min(list_x)),
            np.abs(np.max(list_x)),
            np.abs(np.min(list_y)),
            np.abs(np.max(list_y)),
        )
        # Compute cell centroid
        cell_centroid = centroid(
            list_y,
            list_x,
        )  # (y, x)
        cell_spot = CellSpot(
            frame,
            cell_centroid[1],  # x
            cell_centroid[0],  # y
            -1,
            abs_min_x,
            abs_max_x,
            abs_min_y,
            abs_max_y,
            [[x, y] for x, y in zip(list_x, list_y)],
        )
        cell_spots.append(cell_spot)

    return frame, cell_spots


def signed_area(x: list[float], y: list[float]) -> float:
    """Compute the signed area of a polygon.

    Parameters
    ----------
    x : list[float]
        List of x-coordinates of the polygon.
    y : list[float]
        List of y-coordinates of the polygon.

    Returns
    -------
    float
        Signed area of the polygon.
    """
    n = len(x)
    a = 0.0
    for i in range(n - 1):
        a += x[i] * y[i + 1] - x[i + 1] * y[i]
    return (a + x[n - 1] * y[0] - x[0] * y[n - 1]) / 2.0


def centroid(x: list[float], y: list[float]) -> list[float]:
    """Calculate the centroid of a polygon.

    Parameters
    ----------
    x : list[float]
        List of x-coordinates of the polygon.
    y : list[float]
        List of y-coordinates of the polygon.

    Returns
    -------
    list[float]
        Coordinates of the centroid of the polygon.
    """
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

    return [int(ax / 6.0 / area), int(ay / 6.0 / area)]
