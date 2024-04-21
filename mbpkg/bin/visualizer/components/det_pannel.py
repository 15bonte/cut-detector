from enum import Enum, auto
from typing import Union, Any
from dataclasses import dataclass
from functools import partial

from dash import html, callback, Input, Output, no_update
import dash_mantine_components as dmc

from cut_detector.factories.mid_body_detection_factory import MidBodyDetectionFactory
from cut_detector.factories.mb_support import detection

from mbpkg.detector import Detector

####### Local globals ########

widget_id_to_arg: dict[int, str] = {}

######## Constants ########
SPOT_DETECTION_METHOD = MidBodyDetectionFactory.SPOT_DETECTION_METHOD

class DetMethKind(Enum):
    BIGFISH = auto()
    H_MAX   = auto()
    MM_LOG  = auto()
    MM_DOG  = auto()
    MM_DOH  = auto()

DET_METH_MAPPING = {
    # Str KW
    "bigfish": DetMethKind.BIGFISH,
    "h_maxima": DetMethKind.H_MAX,
    "cur_log": DetMethKind.MM_LOG,
    "lapgau": DetMethKind.MM_LOG,
    "log2_wider": DetMethKind.MM_LOG,
    "rshift_log": DetMethKind.MM_LOG,
    "cur_dog": DetMethKind.MM_DOG,
    "diffgau": DetMethKind.MM_DOG,
    "cur_doh": DetMethKind.MM_DOH,
    "hessian": DetMethKind.MM_DOH,

    # partial.func
    detection.detect_minmax_log: DetMethKind.MM_LOG,
    detection.detect_minmax_dog: DetMethKind.MM_DOG,
    detection.detect_minmax_doh: DetMethKind.MM_DOH,
}

class NoExtraction:
    pass

@dataclass
class PartialExtractor:
    args: list[str]

EXTRACT_PARAM_MAPPING = {
    DetMethKind.BIGFISH: NoExtraction(),
    DetMethKind.H_MAX: NoExtraction(),
    DetMethKind.MM_LOG: PartialExtractor(["min_sigma", "max_sigma", "num_sigma", "threshold"]),
    DetMethKind.MM_DOG: PartialExtractor(["min_sigma", "max_sigma", "sigma_ratio", "threshold"]),
    DetMethKind.MM_DOH: PartialExtractor(["min_sigma", "max_sigma", "num_sigma", "threshold"]),
}

@dataclass
class NoWidgetJustText:
    t: str

@dataclass
class WidgetList:
    l: list

@dataclass
class NumberWidget:
    associated_arg: str
    label: str


METH_WIDGET_MAPPING = {
    DetMethKind.BIGFISH: NoWidgetJustText("No settings available for bigfish"),
    DetMethKind.H_MAX: NoWidgetJustText("No settings available for h_maxima"),
    DetMethKind.MM_LOG: WidgetList([
        NumberWidget("min_sigma", "Minimal Sigma"),
        NumberWidget("max_sigma", "Maximal Sigma"),
        NumberWidget("num_sigma", "Number of Sigmas"),
        NumberWidget("threshold", "Threshold"),
    ]),
    DetMethKind.MM_DOG: WidgetList([
        NumberWidget("min_sigma", "Minimal Sigma"),
        NumberWidget("max_sigma", "Maximal Sigma"),
        NumberWidget("sigma_ratio", "Sigma Ratio"),
        NumberWidget("threshold", "Threshold"),
    ]),
    DetMethKind.MM_DOH: WidgetList([
        NumberWidget("min_sigma", "Minimal Sigma"),
        NumberWidget("max_sigma", "Maximal Sigma"),
        NumberWidget("num_sigma", "Number of Sigmas"),
        NumberWidget("threshold", "Threshold"),
    ]),
}


######## Helpers ########

class DetMethKindError(Exception):
    pass

def get_detection_method_kind(d_method: SPOT_DETECTION_METHOD) -> DetMethKind:
    if callable(d_method) or isinstance(d_method, str):
        key = d_method.func if isinstance(d_method, partial) else d_method
        if (kind := DET_METH_MAPPING.get(key)) is not None:
            return kind
        else:
            raise DetMethKindError(f"Method {key} is not supported")
    else:
        raise DetMethKindError(
            f"Only callable/str are supported, found {type(d_method)}: {d_method} "
        )
    
def extract_parameters(d_method: SPOT_DETECTION_METHOD, kind: DetMethKind) -> dict[str, Any]:
    if (extractor := EXTRACT_PARAM_MAPPING.get(kind)) is not None:
        if isinstance(extractor, NoExtraction):
            # no extraction required
            return {}
        elif isinstance(extractor, PartialExtractor):
            # partial extractor
            extract_partial(d_method, extractor.args)
        else:
            raise DetMethKindError(f"Unknown param extractor {extractor}")
    else:
        raise DetMethKindError(f"Could not find a param extractor for {d_method}")
    
def extract_partial(p: partial, args: list[str]) -> dict[str, Any]:
    assert isinstance(p, partial), f"p not a partial: {p}"
    return {
        k: p.keywords[k]
        for k in args
    }

def generate_widgets(kind: DetMethKind, args: dict[str, Any]) -> list:
    global widget_id_to_arg
    widget_id_to_arg = {}

    if (widget_request := METH_WIDGET_MAPPING.get(kind)) is not None:
        if isinstance(widget_request, NoWidgetJustText):
            return [dmc.Text(widget_request.t)]
        
        elif isinstance(widget_request, WidgetList):
            widgets = []
            for idx, w in enumerate(widget_request.l):
                if isinstance(w, NumberWidget):
                    widget_id_to_arg[idx] = w.associated_arg
                    widgets.append(dmc.NumberInput(
                        id={"type":"det_param", "index":idx},
                        label=w.label
                    ))
    else:
        raise DetMethKindError(f"kind {kind} has not associated widget mapping")
    pass

######## Callbacks ########

@callback(
    Output("det_param_area", "children"),
    Input("det_sel", "value")
)
def update_param_area(det_repr: str) -> dmc.Stack:
    if isinstance(det_repr, str) and len(det_repr) > 0:
        d = Detector(det_repr)
        d_method = d.to_factory()
        kind = get_detection_method_kind(d_method)
        default_parameters = extract_parameters(d_method, kind)
        return dmc.Stack(generate_widgets(kind, default_parameters))

    else:
        return no_update

######## Layout ########
def make_det_pannel() -> dmc.Stack:
    from .shared import _detectors

    select_data = [v.to_str() for v in _detectors]

    return dmc.Stack(
        children=[
            dmc.Select(
                id="det_sel",
                data=select_data,
                value=select_data[0]
            ),
            html.Div(id="det_param_area")
        ],
    )