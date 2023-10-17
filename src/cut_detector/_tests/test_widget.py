import numpy as np

from cut_detector._widget import whole_process


# make_napari_viewer is a pytest fixture that returns a napari viewer object
def test_example_magic_widget(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((100, 100)))

    # Just try to open the widget
    whole_process()
