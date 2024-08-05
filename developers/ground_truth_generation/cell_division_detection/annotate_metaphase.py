import os
from matplotlib import pyplot as plt
from cnn_framework.utils.readers.tiff_reader import TiffReader


def annotate_cells(
    original_folder: str,
    class_folders: list[str],
    max_number_of_files: int,
    need_last_cells: bool,
) -> None:
    """Annotate cells in images.

    Parameters
    ----------
    original_folder : str
        Path to the folder containing the images to annotate.
    class_folders : list[str]
        List of paths to the folders where to save the annotated images.
        For each image, if you type "0" then the image will be stored in the first of these folders.
        Typically: 0 for interphase, 1 for metaphase and 2 for death.
    max_number_of_files : int
        Maximum number of files in class folder.
    need_last_cells : bool
        Whether to annotate the last cells first.
        Used to get more annotations of dead cells, which are less common than the others
        and likely to be at the end of the video.

    """
    # Create folders to save results if they do not exist yet
    for class_folder in class_folders:
        os.makedirs(class_folder, exist_ok=True)
    full_folders = [False for _ in range(len(class_folders))]

    files = os.listdir(original_folder)

    if need_last_cells:
        files.sort(key=lambda x: -int(x.split("_")[-1].split(".")[0]))

    for file_name in files:
        # If all folders have enough images, break
        if full_folders.count(True) == len(full_folders):
            break

        file_path = os.path.join(original_folder, file_name)
        image = TiffReader(file_path).get_processed_image()
        image = image.squeeze()

        # Function to handle key pressing and writing file in .txt accordingly
        def quit_figure(event):
            category = int(event.key)
            class_directory = class_folders[category]

            # Delete file if key is 9 or if class folder is full
            if category == 9 or full_folders[category] is True:
                os.remove(file_path)
            # If class folder is not full, move file to it
            else:
                os.rename(file_path, os.path.join(class_directory, file_name))
                if len(os.listdir(class_directory)) >= max_number_of_files:
                    full_folders[category] = True
            plt.close(event.canvas.figure)

        # Plot image and connect key press event
        nb_channels = image.shape[-1]

        for idx in range(nb_channels):
            plt.subplot(1, nb_channels, idx + 1)
            if idx == 1:
                plt.title(file_name)
            plt.imshow(image[..., idx], cmap="gray")

        plt.gcf().canvas.mpl_connect("key_press_event", quit_figure)

        # Display on whole screen
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()

        plt.show()


if __name__ == "__main__":
    # Parameters
    ORIGINAL_FOLDER = ""
    CLASS_FOLDERS = ["", "", ""]
    MAX_NUMBER_OF_FILES = 100  # Maximum number of files in class folder

    annotate_cells(
        ORIGINAL_FOLDER,
        CLASS_FOLDERS,
        MAX_NUMBER_OF_FILES,
        need_last_cells=False,
    )
