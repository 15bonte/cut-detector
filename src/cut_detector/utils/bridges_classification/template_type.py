from enum import Enum


# Enum for the different types of templates
class TemplateType(Enum):
    """
    Enum to represent the different embeddings for bridges classification.
    """

    ALL = 1
    ALL_WITHOUT_HARALICK = 2
    NB_PEAKS = 3
    PEAKS_AND_INTENSITY = 4
    HARALICK = 5
    ALL_ABLATION_1 = 6
    ALL_ABLATION_2 = 7
    ALL_ABLATION_3 = 8
    ALL_ABLATION_4 = 9
    ALL_ABLATION_5 = 10
    ALL_ABLATION_6 = 11
    ALL_ABLATION_7 = 12
    ALL_ABLATION_8 = 13
    AVERAGE_CIRCLE = 14
