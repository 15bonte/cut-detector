class BoxDimensions:
    """
    A class to store the dimensions of a box.
    """

    def __init__(self, min_x=None, max_x=None, min_y=None, max_y=None):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y

    def is_empty(self):
        return self.min_x is None

    def update(self, min_x, max_x, min_y, max_y):
        if self.min_x is None or min_x < self.min_x:
            self.min_x = min_x
        if self.max_x is None or max_x > self.max_x:
            self.max_x = max_x
        if self.min_y is None or min_y < self.min_y:
            self.min_y = min_y
        if self.max_y is None or max_y > self.max_y:
            self.max_y = max_y
        # Protect against wrong box dimensions
        assert self.min_x <= self.max_x
        assert self.min_y <= self.max_y

    def update_from_box_dimensions(self, box_dimensions):
        self.update(
            box_dimensions.min_x, box_dimensions.max_x, box_dimensions.min_y, box_dimensions.max_y
        )

    def overlaps(self, box_dimensions):
        return (
            self.min_x <= box_dimensions.max_x
            and self.max_x >= box_dimensions.min_x
            and self.min_y <= box_dimensions.max_y
            and self.max_y >= box_dimensions.min_y
        )
