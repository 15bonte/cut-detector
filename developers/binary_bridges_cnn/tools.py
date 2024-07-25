import numpy as np

from cut_detector.utils.tools import apply_hmm


def get_category_from_name(initial_category, mode):
    """
    From 10 initial categories to 3 categories: no cut, cut, two cuts.
    Mode is either microtubules or membrane.
    """
    initial_category = int(initial_category)
    initial_category = initial_category % 5

    if mode == "microtubules":
        if initial_category in [0]:
            category = 0
        elif initial_category in [1, 3]:
            category = 1
        elif initial_category in [2, 4]:
            category = 2

    elif mode == "membrane":
        if initial_category in [0, 1, 2]:
            category = 0
        elif initial_category in [3, 4]:
            category = 1

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return category


def get_first_index(lst, value):
    """
    Get index of first occurrence of given value in list, or any value higher.
    """
    if max(lst) < value:
        return len(lst)
    for i, v in enumerate(lst):
        if v >= value:
            return i
    raise ValueError("Value not found in list")


def evaluate_frame_error(
    predictions,
    names,
    hmm_bridges_parameters_file,
    target_category=1,
):
    # Store categories in a dictionary
    classification = {}
    for name, prediction in zip(names, predictions):
        category = name.split("_c")[1][0]
        core_name, frame = name.split("_c")[0].rsplit("_", 1)
        if core_name not in classification:
            classification[core_name] = {"predicted": {}, "ground_truth": {}}
        classification[core_name]["ground_truth"][int(frame)] = (
            get_category_from_name(category, mode="microtubules")
        )
        classification[core_name]["predicted"][int(frame)] = np.argmax(
            prediction
        )

    # Load HMM model to smooth predictions
    hmm_parameters = np.load(hmm_bridges_parameters_file)

    frame_differences = []
    for mitosis_name, mitosis_classification in classification.items():
        frames = list(mitosis_classification["ground_truth"].keys())
        frames.sort()
        ground_truth = [
            mitosis_classification["ground_truth"][frame] for frame in frames
        ]
        predicted = [
            mitosis_classification["predicted"][frame] for frame in frames
        ]
        corrected_predicted = apply_hmm(hmm_parameters, predicted)
        # Get index of first occurrence of target_category
        target_category_index = get_first_index(ground_truth, target_category)
        predicted_target_category_index = get_first_index(
            corrected_predicted, target_category
        )
        frame_difference = (
            predicted_target_category_index - target_category_index
        )
        if abs(frame_difference) > 5:
            print(f"Frame difference: {frame_difference} on {mitosis_name}")
        frame_differences.append(frame_difference)

    print(
        f"Mean frame difference: {np.mean(frame_differences)} +/- {np.std(frame_differences)} on {len(frame_differences)} mitoses."
    )
