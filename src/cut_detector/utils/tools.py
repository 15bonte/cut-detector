import os
import pickle
import ast
import csv
from typing import Optional
import torch
from torch.utils.data import DataLoader
import numpy as np

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

from .mitosis_track import MitosisTrack
from .cnn_data_set import CnnDataSet
from .hidden_markov_models import HiddenMarkovModel


def re_organize_channels(image: np.ndarray) -> np.ndarray:
    """Expect a 4 dimensions image.
    Re-organize channels to get time as first dimension and channel at last.

    Parameters
    ----------
    image : np.ndarray
        Image to re-organize.

    Returns
    -------
    np.ndarray
        Re-organized image.
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


def get_annotation_file(
    video_path: str, mitosis_track, annotations_files: list[str]
) -> Optional[str]:
    """Get annotation file corresponding to a mitosis track. Return None if not found.

    Parameters
    ----------
    video_path : str
        Path to video file.
    mitosis_track : MitosisTrack
        Mitosis track to find annotation file for.
    annotations_files : list[str]
        List of annotation files.

    Returns
    -------
    Optional[str]
        Path to annotation file.
    """
    video_file = os.path.basename(video_path).split(".")[0]
    mitosis_id = f"mitosis_{mitosis_track.id}_"
    # Assume that mitosis id is consistent across all segmentation/tracking runs
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
    save: bool,
) -> tuple[int]:
    """Function used to upload annotations and perform mid-body detection evaluation.

    Parameters
    ----------
    annotations_folder : str
        Folder containing annotations.
    video_path : str
        Path to video file.
    mitoses_folder : str
        Folder containing mitosis tracks.
    save : bool
        Whether to update mitoses with annotations.

    Returns
    -------
    tuple[int]
        Number of mid-bodies detected and not detected.
    """

    # Read video
    video_name = os.path.basename(video_path).split(".")[0]

    mitosis_tracks: list[MitosisTrack] = []
    # Iterate over "bin" files in exported_mitoses_dir
    for state_path in os.listdir(mitoses_folder):
        # Ignore if not for current video
        if video_name not in state_path:
            continue
        # Load mitosis track
        with open(os.path.join(mitoses_folder, state_path), "rb") as f:
            mitosis_track = MitosisTrack.load(f)

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
            mitosis_track.update_mid_body_ground_truth(annotation_file)

            # Save updated mitosis track
            if save:
                state_path = f"{mitosis_track.get_file_name(video_name)}.bin"
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
        f"\nMid-body detection evaluation: {mb_detected / (mb_detected + mb_not_detected) * 100:.2f}% - {mb_detected}/{mb_detected + mb_not_detected}"
    )

    return mb_detected, mb_not_detected


def upload_annotations_folder(
    annotations_folder: str,
    video_folder: str,
    mitoses_folder: str,
    save: bool,
) -> tuple[int]:
    """Main annotations upload function.

    Parameters
    ----------
    annotations_folder : str
        Folder containing annotations.
    video_path : str
        Path to video folder.
    mitoses_folder : str
        Folder containing mitosis tracks.
    save : Optional[bool], optional
        Whether to update mitoses with annotations, by default True.

    Returns
    -------
    tuple[int]
        Number of mid-bodies detected and not detected.
    """
    all_detected, all_not_detected = 0, 0

    video_paths = os.listdir(video_folder)
    for video_idx, video in enumerate(video_paths):
        video_progress = f"{video} - {video_idx + 1}/{len(video_paths)}"
        print(f"\nVideo {video_progress}...")
        # Perform mid-body detection evaluation
        detected, not_detected = upload_annotations(
            annotations_folder,
            os.path.join(video_folder, video),
            mitoses_folder,
            save=save,
        )
        all_detected += detected
        all_not_detected += not_detected

    return all_detected, all_not_detected


def csv_parameters_to_dict(parameters_path: str) -> dict:
    """Read parameters from a csv file and return them as a dictionary.
    Assume to skip some lines if they do not have the right format.

    Parameters
    ----------
    parameters_path : str
        Path to csv file.

    Returns
    -------
    dict
        Parameters as dictionary.
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
) -> list[int]:
    """Perform CNN inference on a list of images.

    Parameters
    ----------
    model_path : str
        CNN model path.
    images :  list[np.array]
        list[CYX]
    cnn_model_params : BaseModelParams
        Model parameters.
    cnn_data_set : AbstractDataSet
        Dataset to read data.
    cnn_classifier : ResnetClassifier
        CNN classifier.
    model_manager : CnnModelManager
        Model manager.

    Returns
    -------
    predictions : list[int]
        Predicted classes.

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


def apply_hmm(hmm_parameters, sequence) -> list[int]:
    """Correct the sequence of classes using HMM.

    Parameters
    ----------
    hmm_parameters : dict
        HMM parameters.
    sequence : list[int]
        Sequence of classes.

    Returns
    -------
    list[int]
        Corrected sequence of classes.
    """
    # Define observation sequence
    obs_seq = np.asarray(sequence, dtype=np.int32)

    # Define HMM model & run inference
    model = HiddenMarkovModel(
        hmm_parameters["A"], hmm_parameters["B"], hmm_parameters["pi"]
    )
    states_seq, _ = model.viterbi_inference(obs_seq)

    return states_seq


def get_video_path(video_name: str, mitoses_path: str) -> str:
    """Return the path of the video corresponding to the given name.
    NB: used for developers only.

    Parameters
    ----------
    video_name : str
        Name of the video.
    mitoses_path : str
        Path to the mitoses folder.

    Returns
    -------
    str
        Path of the video.
    """
    # Video name start (acquisition name)
    video_name_start = "*" + video_name.split("mitosis")[0].split(" ")[-1]
    # Video name end (x_to_y.tiff)
    video_name_end = "*" + "_".join(video_name.split("_")[-3:])
    video_name_candidates = extract_patterns(
        os.listdir(mitoses_path), [video_name_start + video_name_end]
    )
    # Check if video exists
    if len(video_name_candidates) == 0:
        return None

    # Construct video path
    video_path = os.path.join(mitoses_path, video_name_candidates[0])

    return video_path
