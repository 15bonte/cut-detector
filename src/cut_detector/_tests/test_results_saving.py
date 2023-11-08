from cut_detector._widget import results_saving
from cut_detector.data.tools import get_data_path


def test_open_results_saving_widget():
    # Just try to open the widget
    results_saving()


def test_results_saving_widget():
    # Open the widget
    widget = results_saving()

    # Run process
    widget(get_data_path("mitoses"), get_data_path("results"))


if __name__ == "__main__":
    test_open_results_saving_widget()
    test_results_saving_widget()
