from cut_detector.utils.mb_support import detection

from .ext_api import DetectorExtension, Widget, NumericWidget

from .mm_dog_support import (
    mm_dog_widgets,
    mm_dog_layer_widget_maker,
    mm_dog_param_extractor, 
    mm_dog_layer_param_extractor,
    mm_dog_maker,
    mm_dog_layer_detector_debug
)


###### Extension binding #########

AVAILABLE_DETECTORS: dict[str, DetectorExtension] = {
    "diffgau": DetectorExtension(
        detector=detection.diffgau,

        param_extractor=mm_dog_param_extractor,
        widgets=mm_dog_widgets(),
        detector_maker=mm_dog_maker,

        layer_list=["Sigma Layer"],
        layer_param_list={
            "Sigma Layer": ["sigma", "ratio"]
        },
        layer_param_extractor=mm_dog_layer_param_extractor,
        layer_widget_maker=mm_dog_layer_widget_maker,
        layer_detector_debug=mm_dog_layer_detector_debug,
    ),

    "dog005": DetectorExtension(
        detector=detection.dog_005,
        
        param_extractor=mm_dog_param_extractor,
        widgets=mm_dog_widgets(),
        detector_maker=mm_dog_maker,

        layer_list=["Sigma Layer"],
        layer_param_list={
            "Sigma Layer": ["sigma", "ratio"]
        },
        layer_param_extractor=mm_dog_layer_param_extractor,
        layer_widget_maker=mm_dog_layer_widget_maker,
        layer_detector_debug=mm_dog_layer_detector_debug,
    ),
}


