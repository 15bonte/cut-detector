""" New version of detection pannel
"""
from typing import Union, Any
from dataclasses import dataclass
from functools import partial

from dash import html, callback, Input, Output, ALL, no_update
import dash_mantine_components as dmc

from cut_detector.utils.mb_support import detection

from mbpkg.better_detector import Detector

######## Local globals ########
widget_id_to_arg = {}

######## Constants ########

class Widget:
    @dataclass
    class NumberWidget:
        arg: str
        label: str


class WidgetGenerator:
    @dataclass
    class JustStr:
        s: str

    @dataclass
    class PartialGenerator:
        widgets: list[Widget]

LOG_DOH_GEN = WidgetGenerator.PartialGenerator([
    Widget.NumberWidget("min_sigma", "Minimal Sigma"),
    Widget.NumberWidget("max_sigma", "Max Sigma"),
    Widget.NumberWidget("num_sigma", "Number of Sigmas"),
    Widget.NumberWidget("threshold", "Threshold")
])

DOG_GEN = WidgetGenerator.PartialGenerator([
    Widget.NumberWidget("min_sigma", "Minimal Sigma"),
    Widget.NumberWidget("max_sigma", "Max Sigma"),
    Widget.NumberWidget("sigma_ratio", "Sigma growing ratio"),
    Widget.NumberWidget("threshold", "Threshold")
])

DETECTOR_WIDGET_MAPPING = {
    # KWStr mapping
    "bigfish":    WidgetGenerator.JustStr("No parameters available for bigfish"),
    "h_maxima":   WidgetGenerator.JustStr("No parameters available for h_maxima"),

    "cur_log":    LOG_DOH_GEN,
    "lapgau":     LOG_DOH_GEN,
    "log2_wider": LOG_DOH_GEN,
    "rshift_log": LOG_DOH_GEN,

    "cur_dog": DOG_GEN,
    "diffgau": DOG_GEN,

    "cur_doh": LOG_DOH_GEN,
    "hessian": LOG_DOH_GEN,

    # Callable mapping 
    # (None for now)

    # Partial mapping
    detection.detect_minmax_log: LOG_DOH_GEN,
    detection.detect_minmax_dog: DOG_GEN,
    detection.detect_minmax_doh: LOG_DOH_GEN,
}

######## Callback ########
@callback(
    Output("det_param_area", "children"),
    Input("det_sel", "value")
)
def update_param_area(det_str: str) -> list:
    # det_str is either a Detector's kwstr or a fmtstr
    detector = Detector(det_str)
    func = detector.v.func if isinstance(detector.v, partial) else detector.v
    if (gen := DETECTOR_WIDGET_MAPPING.get(func)) is None:
        raise DetectorPannelError(f"No widget generation strategy for func: {func}")
    
    return generate_widgets(detector, gen)

@callback(
    Output("sig_cur_det", "data"),
    Input("det_sel", "value")
)
def update_sig_cur_detector(selected_detector: str) -> str:
    if isinstance(selected_detector, str) and len(selected_detector) > 0:
        return selected_detector
    else:
        return no_update

@callback(
    Output("sig_det_param", "data"),
    Input({"type": "det_param", "index": ALL}, "value")
)
def update_sig_det_param(widget_values: list[float]) -> dict[str, Any]:
    return {
        widget_id_to_arg[idx]: v
        for idx, v in enumerate(widget_values)
    }

######## Helpers ########
class DetectorPannelError(Exception):
    pass

def generate_widgets(
        detector: Detector, 
        gen: Union[WidgetGenerator.JustStr, WidgetGenerator.PartialGenerator],
        ) -> list:
    global widget_id_to_arg
    if isinstance(gen, WidgetGenerator.JustStr):
        return [dmc.Text(gen.s)]
    elif isinstance(gen, WidgetGenerator.PartialGenerator):
        l = []
        params = extract_available_parameters(detector)
        print("avaiable params:", params)
        for idx, w in enumerate(gen.widgets):
            if isinstance(w, Widget.NumberWidget):
                widget_id_to_arg[idx] = w.arg
                if (default_v := params.get(w.arg)) is None:
                    raise DetectorPannelError(
                        f"Missing argument {w.arg} in detector {detector}"
                    )
                l.append(dmc.NumberInput(
                    id={"type": "det_param", "index":idx},
                    label=w.label,
                    value=default_v
                ))
            else:
                raise DetectorPannelError(f"Invalid widget: {w}")
        return l
    else:
        raise DetectorPannelError(f"Invalid generatrion strategy: {gen}")
    
def extract_available_parameters(d: Detector) -> dict[str, Any]:
    # case 1: if d.v is non-partial callable, then it does not have settable parameters,
    # since its signature must be (np.ndarray) -> np.ndarray
    # so we can try to convert it to a partial and worst case 
    # we get a partial(func=f, keywords={}) which actually is perfect because
    # this explicitely says we don't have any parameters.
    #
    # case 2: if d.v is a partial, we can get its parameters directly
    #
    # case 3: if d.v is a str, then it must be a kwstr. So we can try to convert it
    # to its associated partial (like case number 1). If found we follow case 1
    # If None then it means we don't have any parameters we raise an error
    #
    # conclusion: just call try_to_partial(). If it returns None, it does not have
    # parameters (because it is a kwstr without function).
    # just read p.keywords; this field will be empty for function that were not
    # partial from the start, which is OK).

    if (p := d.try_to_partial()) is not None:
        return p.keywords
    else:
        return {}

######## Layout ########
def make_det_pannel() -> dmc.Stack:
    from .shared import _detectors

    select_data = [s if (s := v.try_to_kwstr()) is not None else v.to_str() for v in _detectors]

    return dmc.Card([
        dmc.CardSection("Detector parameters"),
        dmc.Stack(
            children=[
                dmc.Select(
                    id="det_sel",
                    data=select_data,
                    value=select_data[0]
                ),
                html.Div(id="det_param_area"),
            ],
        )
    ])