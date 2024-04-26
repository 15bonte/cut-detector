from cut_detector.utils.mb_support import detection

from .ext_api import DetectorExtension, Widget, NumericWidget

from .mm_dog_support import mm_dog_widgets, mm_dog_param_extractor, mm_dog_maker


###### Extension binding #########
AVAILABLE_DETECTORS: dict[str, DetectorExtension] = {
    "diffgau": DetectorExtension(
        detection.diffgau,
        mm_dog_param_extractor,
        mm_dog_widgets(),
        mm_dog_maker
    ),

    "dog005": DetectorExtension(
        detection.dog_005,
        mm_dog_param_extractor,
        mm_dog_widgets(),
        mm_dog_maker
    ),
}


