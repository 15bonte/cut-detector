from cut_detector._widget import results_saving
from cut_detector.data.tools import get_data_path
from cut_detector.widget_functions.save_results import perform_results_saving


def test_open_results_saving_widget():
    # Just try to open the widget
    results_saving()


def test_results_saving():
    perform_results_saving(
        get_data_path("mitoses"),
        show=False,
        save_dir=get_data_path("results"),
        verbose=True,
    )


if __name__ == "__main__":
    test_open_results_saving_widget()
    test_results_saving()
