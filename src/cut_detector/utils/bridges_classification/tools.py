import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy.signal import find_peaks
from skimage.feature import graycomatrix, graycoprops
import matplotlib.pyplot as plt

from ...constants.bridges import IMAGE_SIZE
from ...utils.hidden_markov_models import HiddenMarkovModel


from .bridges_classification_parameters import (
    BridgesClassificationParameters,
)
from .template_type import TemplateType


def get_bridge_class(
    img_bridge: np.ndarray,
    scaler: StandardScaler,
    clf: SVC,
    parameters=BridgesClassificationParameters(),
    plot_enabled=True,
) -> int:
    # Make sure crop has expected size
    assert img_bridge.shape[0] == IMAGE_SIZE and img_bridge.shape[1] == IMAGE_SIZE

    # Get template and reshape to 2D data
    template = get_bridge_template(
        img_bridge, parameters=parameters, plot_enabled=plot_enabled
    ).reshape(1, -1)

    # Scale template
    scaled_template = scaler.transform(template)

    # Predict the class
    frame_class = clf.predict(scaled_template)
    if frame_class[0] == "A":
        return 0
    if frame_class[0] == "B":
        return 1
    return 2


def apply_hmm(hmm_parameters, sequence):
    """
    Correct the sequence of classes using HMM.
    """
    # Define observation sequence
    obs_seq = np.asarray(sequence, dtype=np.int32)

    # Define HMM model & run inference
    model = HiddenMarkovModel(hmm_parameters["A"], hmm_parameters["B"], hmm_parameters["pi"])
    states_seq, _ = model.viterbi_inference(obs_seq)

    return states_seq


def get_circle_data(
    radius: int,
    image: np.ndarray,
    parameters: BridgesClassificationParameters,
) -> tuple[list[tuple[int]], list[float], float, np.ndarray, np.ndarray]:
    """
    Compute useful data for a circle of a given radius around the mid body spot:

    - circle_positions: list of the coordinates of the pixels on the circle
    - intensities: list of the intensities of the pixels on the circle
    - mean_intensity: mean intensity of the pixels on the circle
    - peaks: list of the indexes of the peaks of the intensities on the circle
    - norm_peaks: list of the normalized indexes of the peaks of the intensities on the circle
    """
    # Get image shape
    size_x = image.shape[1]
    size_y = image.shape[0]

    # Get the mid-body spot coordinates, middle of the image
    mid_body_position = (size_x // 2, size_y // 2)

    # Get associated perimeter, angles and coordinates
    perimeter = round(2 * np.pi * radius)
    angles = np.linspace(0, 2 * np.pi, perimeter)
    circle_positions = []
    for angle in angles:
        pos_x = round(mid_body_position[1] + radius * np.cos(angle))
        pos_y = round(mid_body_position[0] + radius * np.sin(angle))
        if (pos_x, pos_y) not in circle_positions:
            circle_positions.append((pos_x, pos_y))

    intensities = []
    for position in circle_positions:
        # Get the mean intensity around the mid spot
        total, inc = 0, 0
        for k in range(-1, 2):
            for j in range(-1, 2):
                pos_x = position[0] + k
                pos_y = position[1] + j
                if 0 <= pos_x < size_x and 0 <= pos_y < size_y:
                    total += image[pos_x, pos_y]
                    inc += 1
        mean = total / inc
        intensities.append(mean)

    # Detect the peaks of the intensities on the circle
    mean_intensity = np.mean(intensities)

    # Prevent the case where the peak is on the border so we detect 2 peaks instead of 1
    concatenated_intensities = intensities + intensities

    # Insert lower intensity at the beginning and at the end of the list
    concatenated_intensities.insert(0, concatenated_intensities[0] - 1)
    concatenated_intensities.append(concatenated_intensities[-1] - 1)

    peaks, _ = find_peaks(
        concatenated_intensities,
        height=mean_intensity * parameters.coeff_height_peak,
        distance=len(intensities) * parameters.circle_min_ratio,
        width=(None, round(len(intensities) * parameters.coeff_width_peak)),
    )

    peaks = [i - 1 for i in peaks]
    peaks = [p % len(circle_positions) for p in peaks]
    peaks = [p for p in peaks if peaks.count(p) >= 2]
    peaks = list(set(peaks))

    norm_peaks = [peak / len(circle_positions) for peak in peaks]

    return circle_positions, intensities, mean_intensity, peaks, norm_peaks


def get_peaks(
    all_positions: list[list[tuple[int]]],
    all_intensities: list[list[float]],
    all_peaks: list[np.ndarray],
    all_norm_peaks: list[np.ndarray],
    parameters: BridgesClassificationParameters,
) -> tuple[list[tuple[int]], list[float]]:
    """
    Post-processing of peaks detection.

    Returns kept peaks (circle index, peak position) and their intensities.
    A typical output would be:
        final_peaks = [(0, 30), (0, 62)]
        peaks_intensity = [440, 416]
    Peak at position 30 on circle 0 was kept, with an intensity of 440.
    Peak at position 62 on circle 0 was kept, with an intensity of 416.
    """
    # Create the windows
    windows_starts = np.linspace(
        0, 1 - parameters.window_size, num=round(parameters.overlap / parameters.window_size)
    )
    windows_ends = np.linspace(
        parameters.window_size, 1, num=round(parameters.overlap / parameters.window_size)
    )
    windows = list(zip(windows_starts, windows_ends))

    # Group the peaks by windows
    group_peaks = []  # list (window) of list of tuple (index of the circle, position of the peak)
    for window in windows:
        group = []
        for i, (peaks, norm_peaks) in enumerate(zip(all_peaks, all_norm_peaks)):
            for peak, norm_peak in zip(peaks, norm_peaks):
                if window[0] <= norm_peak < window[1]:
                    group.append((i, peak))
        group_peaks.append(group)

    # Remove groups that have less than min_peaks_by_group peaks
    new_group_peaks = []
    for group_peak in group_peaks:
        if len(group_peak) >= parameters.min_peaks_by_group and group_peak not in new_group_peaks:
            new_group_peaks.append(group_peak)

    # If there is more than 1 peak, make sure there is no duplicate
    if len(new_group_peaks) >= 2:
        # Sort by maximum number of peaks, or higher mean intensity
        def get_length_intensity_key(item):
            return (len(item), np.mean([all_intensities[p[0]][p[1]] for p in item]))

        new_group_peaks.sort(key=get_length_intensity_key, reverse=True)

        # One peak must appear in one group only
        # Remove the peaks that are already seen in previous groups
        for i in range(len(new_group_peaks) - 1):
            for j in range(i + 1, len(new_group_peaks)):
                new_group_peaks[j] = [p for p in new_group_peaks[j] if p not in new_group_peaks[i]]

        # Remove groups that have less than min_peaks_by_group peaks
        new_group_peaks = [g for g in new_group_peaks if len(g) >= parameters.min_peaks_by_group]

        # Compute the mean position of the peaks in each group
        position_groups = [
            sum(pos / len(all_positions[circle]) for circle, pos in group) / len(group)
            for group in new_group_peaks
        ]

        # Remove the groups that are too close to each other, i.e. distance < circle_min_ratio
        i = 0
        while i < len(position_groups) - 1:
            j = i + 1
            while j < len(position_groups):
                first_position = (position_groups[i] - position_groups[j]) % 1
                second_position = (position_groups[j] - position_groups[i]) % 1
                if min(first_position, second_position) < parameters.circle_min_ratio:
                    if len(new_group_peaks[i]) > len(new_group_peaks[j]):
                        new_group_peaks.remove(new_group_peaks[j])
                        position_groups.remove(position_groups[j])
                        j -= 1
                    else:
                        new_group_peaks.remove(new_group_peaks[i])
                        position_groups.remove(position_groups[i])
                        i -= 1
                j += 1
            i += 1

        # Sort again by maximum number of peaks and mean intensity of the peaks in the group
        new_group_peaks.sort(key=get_length_intensity_key, reverse=True)

        # Should not be possible to have more than 2 MT, then keep 2 best ones
        if len(new_group_peaks) > 2:
            new_group_peaks = new_group_peaks[:2]

    # Compute peaks_intensity with the mean intensity of the peaks
    peaks_intensity = [
        sum(all_intensities[circle][pos] for circle, pos in group) / len(group)
        for group in new_group_peaks
    ]

    # final_peaks for plot
    final_peaks = [p[0] for p in new_group_peaks]

    return final_peaks, peaks_intensity


def get_haralick_features(
    list_intensity: list[float],
    properties=None,  # avoid list as default value
) -> np.ndarray:
    """
    Compute a few Haralick features from a list of intensities.
    """
    if properties is None:
        properties = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]

    # Convert list_intensity to 2D array and int
    list_intensity_2D = np.array(list_intensity).astype(int).reshape(1, -1)

    # Compute all haralick features
    haralick_features = graycomatrix(
        list_intensity_2D,
        distances=[1],
        angles=[0],
        levels=max(list_intensity_2D[0]) + 1,
        symmetric=True,
        normed=True,
    )
    return np.array([graycoprops(haralick_features, property)[0][0] for property in properties])


def plot_circles(
    filename,
    image,
    final_peaks: list[tuple[int]],
    all_positions: list[list[tuple[int]]],
    mean_intensity: list[float],
    peaks_intensity: list[float],
    list_intensity: list[float],
    coeff_height_peak: float,
) -> None:
    """
    Plot the image with the circle and the peaks.
    """

    # Plot the intensities of the circle around the mid body spot
    plt.subplot(2, 1, 1)
    plt.title("Number of peaks: " + str(len(final_peaks)))

    plt.plot(range(len(list_intensity)), list_intensity)
    plt.axhline(y=mean_intensity[0] * coeff_height_peak, color="red")
    plt.plot([p[1] for p in final_peaks], peaks_intensity, marker="o", markersize=4, color="red")

    # plot the image without the bridge line
    plt.subplot(2, 2, 3)
    plt.imshow(image)

    # plot the image with the bridge line
    plt.subplot(2, 2, 4)
    plt.imshow(image)

    if len(final_peaks) > 0:
        coord_peaks = [all_positions[p[0]][p[1]] for p in final_peaks]
        if len(final_peaks) == 2:
            plt.plot(
                [coord_peaks[0][1], coord_peaks[1][1]],
                [coord_peaks[0][0], coord_peaks[1][0]],
                color="red",
            )
        elif len(final_peaks) == 1:
            plt.plot(coord_peaks[0][1], coord_peaks[0][0], marker="o", markersize=4, color="red")
    for i in range(len(all_positions)):
        x_cercle = [all_positions[i][k][0] for k in range(len(all_positions[i]))]
        y_cercle = [all_positions[i][k][1] for k in range(len(all_positions[i]))]
        plt.plot(y_cercle, x_cercle, color="green")

    plt.title(filename)
    plt.show()


def get_bridge_template(
    image,
    parameters: BridgesClassificationParameters,
    plot_enabled: bool,
    filename=None,
) -> np.ndarray:
    """
    Return bridge image embedding according to template provided in parameters.
    """

    # Create the coordinates list of the circle around the mid body spot
    # NB: first circle in list is the middle circle
    circle_radius, diff_radius = parameters.circle_radius, parameters.diff_radius
    list_radius = [
        circle_radius,
        circle_radius + 0.25 * diff_radius,
        circle_radius - 0.25 * diff_radius,
        circle_radius + 0.5 * diff_radius,
        circle_radius - 0.5 * diff_radius,
        circle_radius + 0.75 * diff_radius,
        circle_radius - 0.75 * diff_radius,
        circle_radius + diff_radius,
        circle_radius - diff_radius,
    ]

    # Get useful data for each circle
    all_positions, all_intensities, all_mean_intensity = [], [], []
    all_peaks, all_norm_peaks = [], []
    for radius in list_radius:
        positions, intensities, mean_intensity, peaks, norm_peaks = get_circle_data(
            radius, image, parameters
        )
        all_positions.append(positions)
        all_intensities.append(intensities)
        all_mean_intensity.append(mean_intensity)
        all_peaks.append(peaks)
        all_norm_peaks.append(norm_peaks)

    # Get peaks data
    final_peaks, peaks_intensity = get_peaks(
        all_positions, all_intensities, all_peaks, all_norm_peaks, parameters
    )

    # Get useful Haralick features from middle center circle
    haralick_features = get_haralick_features(all_intensities[0])

    # Plot if enabled
    if plot_enabled:
        plot_circles(
            filename,
            image,
            final_peaks,
            all_positions,
            all_mean_intensity,
            peaks_intensity,
            all_intensities[0],
            parameters.coeff_height_peak,
        )

    # Get the template of the bridge
    if parameters.template_type == TemplateType.AllWithoutHaralick:
        return np.array(
            [
                len(final_peaks),  # 0 : number of peaks
                np.mean(peaks_intensity)
                if len(final_peaks) > 0
                else np.mean(all_intensities[0]),  # 1 : intensity of the peaks
                np.var(all_intensities[0]),  # 2 : variance of intensity on the circle
                np.mean(all_intensities[0]),  # 3 : mean intensity on the circle
            ]
        )

    if parameters.template_type == TemplateType.NbPeaks:
        return np.array(
            [
                len(final_peaks),  # 0 : number of peaks
            ]
        )

    if parameters.template_type == TemplateType.PeaksAndIntensity:
        return np.array(
            [
                len(final_peaks),  # 0 : number of peaks
                np.mean(peaks_intensity)
                if len(final_peaks) > 0
                else np.mean(all_intensities[0]),  # 1 : intensity of the peaks
            ]
        )

    if parameters.template_type == TemplateType.Haralick:
        return haralick_features

    if parameters.template_type == TemplateType.All:
        template = np.array(
            [
                len(final_peaks),  # 0 : number of peaks
                np.mean(peaks_intensity)
                if len(final_peaks) > 0
                else np.mean(all_intensities[0]),  # 1 : intensity of the peaks
                np.mean(all_intensities[0]),  # 2 : mean intensity on the circle
            ]
        )
        return np.concatenate((template, haralick_features))

    # Ablation study on All
    if parameters.template_type == TemplateType.All_Ablation1:
        template = np.array(
            [
                np.mean(peaks_intensity)
                if len(final_peaks) > 0
                else np.mean(all_intensities[0]),  # 1 : intensity of the peaks
                np.mean(all_intensities[0]),  # 2 : mean intensity on the circle
            ]
        )
        return np.concatenate((template, haralick_features))

    if parameters.template_type == TemplateType.All_Ablation2:
        template = np.array(
            [
                len(final_peaks),  # 0 : number of peaks
                np.mean(all_intensities[0]),  # 1 : mean intensity on the circle
            ]
        )
        return np.concatenate((template, haralick_features))

    if parameters.template_type == TemplateType.All_Ablation3:
        template = np.array(
            [
                len(final_peaks),  # 0 : number of peaks
                np.mean(peaks_intensity)
                if len(final_peaks) > 0
                else np.mean(all_intensities[0]),  # 1 : intensity of the peaks
            ]
        )
        return np.concatenate((template, haralick_features))

    if parameters.template_type == TemplateType.All_Ablation4:
        template = np.array(
            [
                len(final_peaks),  # 0 : number of peaks
                np.mean(peaks_intensity)
                if len(final_peaks) > 0
                else np.mean(all_intensities[0]),  # 1 : intensity of the peaks
                np.mean(all_intensities[0]),  # 2 : mean intensity on the circle
            ]
        )
        return np.concatenate((template, haralick_features[1:]))

    if parameters.template_type == TemplateType.All_Ablation5:
        template = np.array(
            [
                len(final_peaks),  # 0 : number of peaks
                np.mean(peaks_intensity)
                if len(final_peaks) > 0
                else np.mean(all_intensities[0]),  # 1 : intensity of the peaks
                np.mean(all_intensities[0]),  # 2 : mean intensity on the circle
            ]
        )
        return np.concatenate((template, haralick_features[:1], haralick_features[2:]))

    if parameters.template_type == TemplateType.All_Ablation6:
        template = np.array(
            [
                len(final_peaks),  # 0 : number of peaks
                np.mean(peaks_intensity)
                if len(final_peaks) > 0
                else np.mean(all_intensities[0]),  # 1 : intensity of the peaks
                np.mean(all_intensities[0]),  # 2 : mean intensity on the circle
            ]
        )
        return np.concatenate((template, haralick_features[:2], haralick_features[3:]))

    if parameters.template_type == TemplateType.All_Ablation7:
        template = np.array(
            [
                len(final_peaks),  # 0 : number of peaks
                np.mean(peaks_intensity)
                if len(final_peaks) > 0
                else np.mean(all_intensities[0]),  # 1 : intensity of the peaks
                np.mean(all_intensities[0]),  # 2 : mean intensity on the circle
            ]
        )
        return np.concatenate((template, haralick_features[:3], haralick_features[4:]))

    if parameters.template_type == TemplateType.All_Ablation8:
        template = np.array(
            [
                len(final_peaks),  # 0 : number of peaks
                np.mean(peaks_intensity)
                if len(final_peaks) > 0
                else np.mean(all_intensities[0]),  # 1 : intensity of the peaks
                np.mean(all_intensities[0]),  # 2 : mean intensity on the circle
            ]
        )
        return np.concatenate((template, haralick_features[:4], haralick_features[5:]))

    raise ValueError("Unknown template type")
