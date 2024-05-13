import numpy as np
import matplotlib.pyplot as plt

from bigfish import stack
from bigfish.plot.utils import save_plot, get_minmax_values
from bigfish.plot.plot_images import _define_patch


def plot_detection(
    image,
    spots,
    shape="circle",
    radius=3,
    color="red",
    linewidth=1,
    fill=False,
    rescale=False,
    contrast=False,
    title=None,
    framesize=(15, 10),
    remove_frame=True,
    path_output=None,
    ext="png",
    show=True,
    add_coord: bool = True,
):
    """NB: most of this function is copied from the bigfish package.

    Plot detected spots and foci on a 2-d image.

    Parameters
    ----------
    image : np.ndarray
        A 2-d image with shape (y, x).
    spots : list or np.ndarray
        Array with coordinates and shape (nb_spots, 3) or (nb_spots, 2). To
        plot different kind of detected spots with different symbols, use a
        list of arrays.
    shape : list or str, default='circle'
        List of symbols used to localized the detected spots in the image,
        among `circle`, `square` or `polygon`. One symbol per array in `spots`.
        If `shape` is a string, the same symbol is used for every elements of
        'spots'.
    radius : list or int or float, default=3
        List of yx radii of the detected spots, in pixel. One radius per array
        in `spots`. If `radius` is a scalar, the same value is applied for
        every elements of `spots`.
    color : list or str, default='red'
        List of colors of the detected spots. One color per array in `spots`.
        If `color` is a string, the same color is applied for every elements
        of `spots`.
    linewidth : list or int, default=1
        List of widths or width of the border symbol. One integer per array
        in `spots`. If `linewidth` is an integer, the same width is applied
        for every elements of `spots`.
    fill : list or bool, default=False
        List of boolean to fill the symbol of the detected spots. If `fill` is
        a boolean, it is applied for every symbols.
    rescale : bool, default=False
        Rescale pixel values of the image (made by default in matplotlib).
    contrast : bool, default=False
        Contrast image.
    title : str, optional
        Title of the image.
    framesize : tuple, default=(15, 10)
        Size of the frame used to plot with ``plt.figure(figsize=framesize)``.
    remove_frame : bool, default=True
        Remove axes and frame.
    path_output : str, optional
        Path to save the image (without extension).
    ext : str or list, default='png'
        Extension used to save the plot. If it is a list of strings, the plot
        will be saved several times.
    show : bool, default=True
        Show the figure or not.

    """
    # check parameters
    stack.check_array(
        image,
        ndim=2,
        dtype=[
            np.uint8,
            np.uint16,
            np.int32,
            np.int64,
            np.float32,
            np.float64,
        ],
    )
    stack.check_parameter(
        spots=(list),
        shape=(list, str),
        radius=(list, int, float),
        color=(list, str),
        linewidth=(list, int),
        fill=(list, bool),
        rescale=bool,
        contrast=bool,
        title=(str, type(None)),
        framesize=tuple,
        remove_frame=bool,
        path_output=(str, type(None)),
        ext=(str, list),
        show=bool,
    )

    # enlist and format parameters
    n = len(spots)
    if not isinstance(shape, list):
        shape = [shape] * n
    elif isinstance(shape, list) and len(shape) != n:
        raise ValueError(
            "If 'shape' is a list, it should have the same "
            "number of items than spots ({0}).".format(n)
        )
    if not isinstance(radius, list):
        radius = [radius] * n
    elif isinstance(radius, list) and len(radius) != n:
        raise ValueError(
            "If 'radius' is a list, it should have the same "
            "number of items than spots ({0}).".format(n)
        )
    if not isinstance(color, list):
        color = [color] * n
    elif isinstance(color, list) and len(color) != n:
        raise ValueError(
            "If 'color' is a list, it should have the same "
            "number of items than spots ({0}).".format(n)
        )
    if not isinstance(linewidth, list):
        linewidth = [linewidth] * n
    elif isinstance(linewidth, list) and len(linewidth) != n:
        raise ValueError(
            "If 'linewidth' is a list, it should have the same "
            "number of items than spots ({0}).".format(n)
        )
    if not isinstance(fill, list):
        fill = [fill] * n
    elif isinstance(fill, list) and len(fill) != n:
        raise ValueError(
            "If 'fill' is a list, it should have the same "
            "number of items than spots ({0}).".format(n)
        )

    # plot
    fig, ax = plt.subplots(1, 2, sharex="col", figsize=framesize)

    # image
    if not rescale and not contrast:
        vmin, vmax = get_minmax_values(image)
        ax[0].imshow(image, vmin=vmin, vmax=vmax)
    elif rescale and not contrast:
        ax[0].imshow(image)
    else:
        if image.dtype not in [np.int64, bool]:
            image = stack.rescale(image, channel_to_stretch=0)
        ax[0].imshow(image, cmap="gray")

    # spots
    if not rescale and not contrast:
        vmin, vmax = get_minmax_values(image)
        ax[1].imshow(image, vmin=vmin, vmax=vmax)
    elif rescale and not contrast:
        ax[1].imshow(image)
    else:
        if image.dtype not in [np.int64, bool]:
            image = stack.rescale(image, channel_to_stretch=0)
        ax[1].imshow(image, cmap="gray")

    for i, coordinates_2d in enumerate(spots):
        # plot symbols
        patch = _define_patch(
            coordinates_2d[1],
            coordinates_2d[0],
            shape[i],
            radius[i],
            color[i],
            linewidth[i],
            fill[i],
        )
        ax[1].add_patch(patch)

        x = int(coordinates_2d[1])
        y = int(coordinates_2d[0])
        ax[1].text(x, y, f"({x} {y})", fontsize=12, color="black")

    # titles and frames
    if title is not None:
        ax[0].set_title(title, fontweight="bold", fontsize=10)
        ax[1].set_title("Detection results", fontweight="bold", fontsize=10)
    if remove_frame:
        ax[0].axis("off")
        ax[1].axis("off")
    plt.tight_layout()

    # output
    if path_output is not None:
        save_plot(path_output, ext)
    if show:
        plt.show()
    else:
        plt.close()
