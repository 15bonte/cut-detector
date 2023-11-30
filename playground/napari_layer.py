import numpy as np
import napari


def main():
    # Create a Napari viewer
    viewer = napari.Viewer()

    # Create a blue rectangle
    rectangle_width = 50
    rectangle_height = 30
    rectangle_color = [0, 0, 1]  # Blue color in RGB

    # Define the rectangle vertices
    vertices = np.array(
        [
            [0, 0],
            [rectangle_width, 0],
            [rectangle_width, rectangle_height],
            [0, rectangle_height],
        ]
    )

    # Create a rectangle layer
    viewer.add_shapes(
        data=vertices,
        shape_type="rectangle",
        edge_color="transparent",
        face_color=rectangle_color,
    )

    # Display the Napari viewer
    napari.run()


if __name__ == "__main__":
    main()
