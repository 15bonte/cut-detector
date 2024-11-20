"""Video parameters for the cut detector."""


class Parameters:
    """
    Class to store video parameters for the cut detector.

    Attributes
    ----------
    spatial_resolution : int
        Spatial resolution of the video.
        In nanometers per pixel, for example 100 means 1 pixel is 100 nm.
    time_resolution : int
        Time resolution of the video.
        In minutes per frame, for example 10 means 1 frame every 10 minutes.
    """

    def __init__(self, spatial_resolution=225, time_resolution=10):
        self.spatial_resolution = spatial_resolution
        self.time_resolution = time_resolution

        # Range to look for metaphase candidate spots
        metaphase_interval = 100  # minutes
        self.frames_around_metaphase = int(
            metaphase_interval / time_resolution
        )  # frames

        # Range to look for midbody in between cells
        cytokinesis_interval = 200  # minutes
        self.cytokinesis_duration = int(
            cytokinesis_interval / time_resolution
        )  # frames

        # Channels
        self.mid_body_channel = 1
        self.sir_channel = 0

        # Classes indexes
        self.interphase_index = 0
        self.metaphase_index = 1
