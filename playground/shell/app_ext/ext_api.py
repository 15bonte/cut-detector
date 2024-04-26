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

    def generate_from_params(self, params: dict[str, Any]) -> html.Div:
        raise RuntimeError("Must be implemented")

@dataclass
class NumericWidget(Widget):
    label: str

    def generate_from_params(self, params: dict[str, Any]) -> dmc.NumberInput:
        return dmc.NumberInput(
            id={"type": "detector_widget", "id": self.id},
            label=self.label,
            value=params[self.fn_arg]
        )
    

@dataclass
class DetectorExtension:
    detector: SPOT_DETECTION_METHOD
    param_extractor: Callable[[SPOT_DETECTION_METHOD], dict[str, Any]]
    widgets: list[Widget]
    detector_maker: Callable[[dict[str, Any]], SPOT_DETECTION_METHOD]

    def generate_and_bind_widgets(self, id_dict: dict[int, str]) -> list:
        params = self.param_extractor(self.detector)
        l = []
        for w in self.widgets:
            w.register_to_id_dict(id_dict)
            l.append(w.generate_from_params(params))
        return l
    
    def make_detector(self, params: dict[str, Any]) -> SPOT_DETECTION_METHOD:
        return self.detector_maker(params)
    
    def initialize_param_dict(self, p_dict: dict[str, Any]):
        p_dict = {}
        params = self.param_extractor(self.detector)
        for k, v in params.items():
            p_dict[k] = v
