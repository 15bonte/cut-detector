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
