from .box_dimensions import BoxDimensions


class BoxDimensionsDln(BoxDimensions):
    """A class to store the dimensions of a box and the delaunay triangulation of the box."""

    def __init__(self):
        super().__init__()
        self.dln = None
        self.list_dln = []

    def get_list_dln(self):
        """Protection against BoxDimensionsDln old version."""
        if hasattr(self, "list_dln"):
            return self.list_dln
        return [self.dln]
