from .box_dimensions import BoxDimensions


class BoxDimensionsDln(BoxDimensions):
    """
    A class to store the dimensions of a box and the delaunay triangulation of the box.
    """

    def __init__(self):
        super().__init__(None, None, None, None)
        self.dln = None
