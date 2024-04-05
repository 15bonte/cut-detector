import os
import pickle
from typing import Optional
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import ast
import csv

import bigfish.stack as stack
from bigfish.plot.utils import save_plot, get_minmax_values
from bigfish.plot.plot_images import _define_patch

from cnn_framework.utils.tools import extract_patterns
from cnn_framework.utils.display_tools import display_progress
from cnn_framework.utils.model_managers.cnn_model_manager import (
    CnnModelManager,
)
from cnn_framework.utils.data_managers.default_data_manager import (
    DefaultDataManager,
)
from cnn_framework.utils.metrics.classification_accuracy import (
    ClassificationAccuracy,
)
from cnn_framework.utils.data_loader_generators.data_loader_generator import (
    collate_dataset_output,
)
from cnn_framework.utils.enum import PredictMode
from cnn_framework.utils.models.resnet_classifier import ResnetClassifier
from cnn_framework.utils.model_params.base_model_params import BaseModelParams

from .cnn_data_set import CnnDataSet
from .hidden_markov_models import HiddenMarkovModel


def re_organize_channels(image: np.ndarray) -> np.ndarray:
    """
    Expect a 4 dimensions image.
    Re-organize channels to get TXYC order.
    """
    if image.ndim != 4:
        raise ValueError("Expect a 4 dimensions image.")

    # Get dimension index of smallest dimension, i.e. channels
    channels_dimension_index = np.argmin(image.shape)
    channels_dimension = image.shape[channels_dimension_index]
    if channels_dimension != 3:
        raise ValueError(
            "Expect 3 channels: SiR-tubulin/MKLP1/Phase contrast, in that order."
        )
    # Put channels at the back
    image = np.moveaxis(image, channels_dimension_index, 3)

    # Get dimension index of second smallest dimension, i.e. time
    second_dimension_index = np.argsort(image.shape)[1]
    # Put time at the front
    image = np.moveaxis(image, second_dimension_index, 0)

    return image


def cell_counter_frame_to_video_frame(
    cell_counter_frame: int, nb_channels=4
) -> int:
    """
    Cell counter index starts at 1, just like Fiji.

    To count frame, it just concatenates all channels.
    For example, with 4 channels, frames 1, 2, 3 and 4 will be frame 1,
    frames 5, 6, 7 and 8 will be frame 2, etc.
    """
    return (cell_counter_frame - 1) // nb_channels


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


def get_annotation_file(video_path, mitosis_track, annotations_files):
    video_file = os.path.basename(video_path).split(".")[0]
    mitosis_id = f"mitosis_{mitosis_track.id}_"
    # Assume that mitosis id is consistent across all Trackmate runs
    # (i.e. possible to use former annotations with new mitoses)
    annotation_path_candidates = extract_patterns(
        annotations_files, ["*" + video_file + "*" + mitosis_id + "*"]
    )
    # Check if video exists
    if len(annotation_path_candidates) == 0:
        return None

    return annotation_path_candidates[0]


def upload_annotations(
    annotations_folder: str,
    video_path: str,
    mitoses_folder: str,
    update_mitoses: Optional[bool] = True,
) -> tuple[int]:
    """
    Function used to upload annotations and perform mid-body detection evaluation.
    """

    # Read video
    video_name = os.path.basename(video_path).split(".")[0]

    mitosis_tracks = []
    # Iterate over "bin" files in exported_mitoses_dir
    for state_path in os.listdir(mitoses_folder):
        # Ignore if not for current video
        if video_name not in state_path:
            continue
        # Load mitosis track
        with open(os.path.join(mitoses_folder, state_path), "rb") as f:
            mitosis_track = pickle.load(f)
            mitosis_track.adapt_deprecated_attributes()

        # Add mitosis track to list
        mitosis_tracks.append(mitosis_track)

    # Get all files in annotations_folder
    annotations_files = []
    for root, _, files in os.walk(annotations_folder):
        for file in files:
            if ".xml" in file:
                annotations_files.append(os.path.join(root, file))

    # Used to evaluate mid-body detection
    mb_detected, mb_not_detected = 0, 0

    # Generate movie for each mitosis and save
    wrong_detections = []
    for i, mitosis_track in enumerate(mitosis_tracks):
        # If annotation is available, load it and update mitosis track
        annotation_file = get_annotation_file(
            video_path, mitosis_track, annotations_files
        )
        if annotation_file is not None:
            mitosis_track.update_mid_body_ground_truth(
                annotation_file, nb_channels=4
            )

            # Save updated mitosis track
            if update_mitoses:
                daughter_track_ids = ",".join(
                    [str(d) for d in mitosis_track.daughter_track_ids]
                )
                state_path = f"{video_name}_mitosis_{mitosis_track.id}_{mitosis_track.mother_track_id}_to_{daughter_track_ids}.bin"
                save_path = os.path.join(
                    mitoses_folder,
                    state_path,
                )
                with open(save_path, "wb") as f:
                    pickle.dump(mitosis_track, f)

            # Evaluate mid-body detection (ignore triple divisions)
            if len(mitosis_track.daughter_track_ids) == 1:
                (
                    is_correctly_detected,
                    percent_detected,
                    average_position_difference,
                ) = mitosis_track.evaluate_mid_body_detection()
                if is_correctly_detected:
                    mb_detected += 1
                else:
                    mb_not_detected += 1
                    wrong_detections.append(
                        {
                            "path": state_path,
                            "percent_detected": percent_detected,
                            "average_position_difference": average_position_difference,
                        }
                    )
                    mitosis_track.evaluate_mid_body_detection()

        display_progress(
            "Update with annotations and evaluate",
            i + 1,
            len(mitosis_tracks),
            additional_message=f"Image {i+1}/{len(mitosis_tracks)}",
        )

    # Print wrong detections
    if len(wrong_detections) > 0:
        print("\nWrong detections:")
        for wrong_detection in wrong_detections:
            print(
                f"{wrong_detection['path']}: detected {wrong_detection['percent_detected']}% with avg distance {wrong_detection['average_position_difference']}"
            )

    if (mb_detected + mb_not_detected) == 0:
        print("No mid-body detection evaluation possible.")
        return 0, 0

    print(
        f"\nMid-body detection evaluation: {mb_detected / (mb_detected + mb_not_detected) * 100:.2f}%"
    )

    return mb_detected, mb_not_detected


def csv_parameters_to_dict(parameters_path: str):
    """
    Read parameters from a csv file and return them as a dictionary.
    Assume to skip some lines if they do not have the right format.
    """
    result_dict = {}
    with open(parameters_path, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for row in csv_reader:
            splitted_row = row[0].split(";", 1)
            if len(splitted_row) == 1:  # if there is no data in second column
                continue
            if len(row) > 1:  # if there is a comma in second column
                splitted_row[1] += "," + ",".join(row[1:])
            key, value = splitted_row[0], splitted_row[1]
            if (
                len(value) and value[0] == "[" and value[-1] == "]"
            ):  # transform string to list
                value = ast.literal_eval(value)
            result_dict[key] = value
    return result_dict


def perform_cnn_inference(
    model_path: str,
    images: list[np.array],
    cnn_model_params: BaseModelParams,
    cnn_data_set=CnnDataSet,
    cnn_classifier=ResnetClassifier,
    model_manager=CnnModelManager,
):
    """
    Perform CNN inference on a list of images.

    Parameters
    ----------
    model_path : str
        CNN model path
    images :  list[np.array]
        list[CYX]
    cnn_model_params : BaseModelParams
        model parameters
    cnn_data_set : AbstractDataSet
        dataset to read data
    cnn_classifier : ResnetClassifier
        CNN classifier
    model_manager : CnnModelManager
        model manager

    Returns
    -------
    predictions : list[int]
        predicted classes

    """
    # Metaphase model parameters
    model_parameters = cnn_model_params()
    # Modify parameters for inference
    model_parameters.train_ratio = 0
    model_parameters.val_ratio = 0
    model_parameters.test_ratio = 1
    model_parameters.models_folder = model_path

    # Read csv parameters file
    parameters_file = os.path.join(model_path, "parameters.csv")
    training_parameters = csv_parameters_to_dict(parameters_file)

    # Load pretrained model
    model = cnn_classifier(
        nb_classes=int(training_parameters["nb_classes"]),
        nb_input_channels=len(training_parameters["c_indexes"])
        * len(training_parameters["z_indexes"]),
        encoder_name=training_parameters["encoder_name"],
    )

    map_location = None
    if not torch.cuda.is_available():
        map_location = torch.device("cpu")
        print("No GPU found, using CPU.")
    model.load_state_dict(
        torch.load(
            os.path.join(model_path, f"{model_parameters.name}.pt"),
            map_location=map_location,
        )
    )

    # Test (no sampler to keep order)
    dataset_test = cnn_data_set(
        images,
        is_train=False,
        names=[f"{idx}.ext" for idx in range(len(images))],
        data_manager=DefaultDataManager(),
        params=model_parameters,
    )
    test_dl = DataLoader(
        dataset_test,
        batch_size=model_parameters.batch_size,
        collate_fn=collate_dataset_output,
    )

    manager = model_manager(model, model_parameters, ClassificationAccuracy)

    predictions = manager.predict(
        test_dl,
        predict_mode=PredictMode.GetPrediction,
        nb_images_to_save=0,
    )  # careful, this is scores and not probabilities
    predictions = [int(np.argmax(p)) for p in predictions]

    return predictions


def apply_hmm(hmm_parameters, sequence):
    """
    Correct the sequence of classes using HMM.
    """
    # Define observation sequence
    obs_seq = np.asarray(sequence, dtype=np.int32)

    # Define HMM model & run inference
    model = HiddenMarkovModel(
        hmm_parameters["A"], hmm_parameters["B"], hmm_parameters["pi"]
    )
    states_seq, _ = model.viterbi_inference(obs_seq)

    return states_seq
