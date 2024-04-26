from dataclasses import dataclass
from typing import Callable, Any, Union

from dash import html
import dash_mantine_components as dmc

from cut_detector.factories.mid_body_detection_factory import MidBodyDetectionFactory
SPOT_DETECTION_METHOD = MidBodyDetectionFactory.SPOT_DETECTION_METHOD

@dataclass
class Widget:
    id: int
    fn_arg: str
    def register_to_id_dict(self, id_dict: dict[int, str]):
        id_dict[self.id] = self.fn_arg

    def generate_from_params(self, params: dict[str, Any], kind: str = "detector_widget") -> html.Div:
        raise RuntimeError("Must be implemented")

@dataclass
class NumericWidget(Widget):
    label: str

    def generate_from_params(self, params: dict[str, Any], kind: str = "detector_widget") -> dmc.NumberInput:
        """generated with default values contained in params"""
        return dmc.NumberInput(
            id={"type": kind, "id": self.id},
            label=self.label,
            value=params[self.fn_arg]
        )
    

@dataclass
class DetectorExtension:
    detector: SPOT_DETECTION_METHOD

    param_extractor: Callable[[SPOT_DETECTION_METHOD], dict[str, Any]]
    widgets: list[Widget]
    detector_maker: Callable[[dict[str, Any]], SPOT_DETECTION_METHOD]

    layer_list: list[str]
    layer_param_extractor: Callable[[SPOT_DETECTION_METHOD, str], dict[str, Any]]
    layer_widget_maker: Callable[[str], list[Widget]]
    
    
    
    def generate_and_bind_widgets(self, id_dict: dict[int, str]) -> list:
        params = self.param_extractor(self.detector)
        l = []
        for w in self.widgets:
            w.register_to_id_dict(id_dict)
            l.append(w.generate_from_params(params))
        return l
    
    def generate_and_bind_layer_widgets(self, layer: str, layer_id_dict: dict[int, str]) -> list:
        params = self.layer_param_extractor(self.detector, layer)
        l = []
        for w in self.layer_widget_maker(layer):
            w.register_to_id_dict(layer_id_dict)
            l.append(w.generate_from_params(params, kind="layer_widget"))
        return l
    
    def initialize_param_dict(self, p_dict: dict[str, Any]):
        p_dict = {}
        params = self.param_extractor(self.detector)
        for k, v in params.items():
            p_dict[k] = v

    def initialize_layer_param_dict(self, layer: str, layer_p_dict: dict[str, Any]):
        layer_p_dict = {}
        params = self.layer_param_extractor(self.detector, layer)
        for k, v in params.items():
            layer_p_dict[k] = v
    
    def make_detector(self, params: dict[str, Any]) -> SPOT_DETECTION_METHOD:
        return self.detector_maker(params)
