import json
from functools import partial
from typing import Optional, Union, Callable, Literal, get_args

import numpy as np

from cut_detector.factories.mid_body_detection_factory import MidBodyDetectionFactory
from cut_detector.factories.mb_support import detection


class DetectorError(Exception):
    pass

class Detector:
    v: MidBodyDetectionFactory.SPOT_DETECTION_METHOD
    repr: Optional[str]
    kind: int

    KnownBlobLikeFn = [
        detection.detect_minmax_log,
        detection.detect_minmax_dog,
        detection.detect_minmax_doh
    ]
    KnownKwStr = get_args(
        get_args(MidBodyDetectionFactory.SPOT_DETECTION_METHOD)[1]
    )

    KnownBlobLikeKind  = 0
    CustomBlobLikeKind = 1
    KwStrKind          = 2
    FmtStrKind         = 3
    FmtDictKind        = 4

    class SpotMethodKind:
        log      = 0
        dog      = 1
        doh      = 2
        h_maxima = 3
        bigfish  = 4

    def __init__(
            self, 
            v: Union[str, Callable[[np.ndarray], np.ndarray], dict], 
            override_repr: Optional[str] = None):
        """ Create a Detector from:
        - a recognized blob-like callable
        - a keyword str (like cur_log, diffgau, etc...)
        - a one-line JSON str 
        - a specially formatted dict: a field 'kind' with the name of
          a recognized function kind, and its arguments fields to use in
          the partial call.

        If you are passing directly an unrecognized blob-like function,
        you will also have to define 'override_repr' with a custom name.

        override_repr is not used in any other case.
        """

        if isinstance(v, partial) and v.func in Detector.KnownBlobLikeFn:
            # Recognized callable
            self.v = v
            self.repr = make_json_str_from_partial(v)
            self.kind = Detector.KnownBlobLikeKind

        elif callable(v):
            # callable but not recognized, custom str required
            if override_repr is not None:
                self.v = v
                self.repr = override_repr
                self.kind = Detector.CustomBlobLikeKind
            else:
                raise DetectorError("Init with unknown callable must define an override_repr")
            
        elif v in Detector.KnownKwStr:
            # known kw str
            self.v = v
            self.repr = v
            self.kind = Detector.KwStrKind
            
        elif isinstance(v, str):
            # specially formatted JSON str
            d: dict = json.loads(v)
            kind = d["kind"]
            d.pop("kind")
            
            self.kind = Detector.FmtStrKind
            self.v = make_known_partial(kind, d)
            self.repr = make_json_str_from_partial(v)

        elif isinstance(v, dict):
            # specially formatted dict
            kind = v["kind"]
            args = v.copy()
            args.pop("kind")

            self.kind = Detector.FmtDictKind
            self.v = make_known_partial(kind, args)
            self.repr = make_json_str_from_partial(self.v)

        else:
            # error: invalid v
            raise DetectorError(f"invalid v: {v}")
    

    def to_factory(self) -> MidBodyDetectionFactory.SPOT_DETECTION_METHOD:
        return self.v
    
    def to_str(self) -> str:
        return self.repr
    
    def get_spot_method_kind(self) -> int:
        MidBodyDetectionFactory.SPOT_DETECTION_METHOD
        if isinstance(self.v, str):
            str_mapping = {
                "bigfish":    Detector.SpotMethodKind.bigfish,
                "h_maxima":   Detector.SpotMethodKind.h_maxima,
                "cur_log":    Detector.SpotMethodKind.log,
                "lapgau":     Detector.SpotMethodKind.log,
                "log2_wider": Detector.SpotMethodKind.log,
                "rshift_log": Detector.SpotMethodKind.log,
                "cur_dog":    Detector.SpotMethodKind.dog,
                "diffgau":    Detector.SpotMethodKind.dog,
                "cur_doh":    Detector.SpotMethodKind.doh,
                "hessian":    Detector.SpotMethodKind.doh,
            }
            if (m := str_mapping.get(self.v)) is not None:
                return m
            else:
                raise DetectorError(f"Unrecognized str method {self.v}")
        elif isinstance(self.v, partial):
            mapping = {
                detection.detect_minmax_log: Detector.SpotMethodKind.log,
                detection.detect_minmax_dog: Detector.SpotMethodKind.dog,
                detection.detect_minmax_doh: Detector.SpotMethodKind.doh,
            }
            if (m := mapping.get(self.v.func)) is not None:
                return m
            else:
                raise Detector(f"Unsupported callable func {self.v.func}")
        else:
            raise DetectorError(f"Unsupported spot method {self.v}")
        
    
    def get_partial(self) -> partial:
        if isinstance(self.v, partial):
            return self.v
        elif isinstance(self.v, str):
            available_callables = {
                "cur_log":    detection.cur_log,
                "lapgau":     detection.lapgau,
                "log2_wider": detection.log2_wider,
                "rshift_log": detection.rshift_log,
                "cur_dog":    detection.cur_dog,
                "diffgau":    detection.diffgau,
                "cur_doh":    detection.cur_doh,
                "hessian":    detection.hessian,
            }
            if(c := available_callables.get(self.v)) is not None:
                return c
            else:
                raise DetectorError(f"No partial available for str {self.v}")
        else:
            raise DetectorError(f"Unsupported v: {self.v}")

    
    def __str__(self) -> str:
        return self.repr
    
    def __repr__(self) -> str:
        return self.repr
    

def make_known_partial(kind: Literal["log", "dog", "doh"], args: dict) -> partial:
    mapping = {
        "log": detection.detect_minmax_log,
        "dog": detection.detect_minmax_dog,
        "doh": detection.detect_minmax_doh
    }
    if (fn := mapping.get(kind)) is not None:
        return partial(
            fn,
            **args
        )
    else:
        raise DetectorError(f"cannot make partial from kind {kind}")


def make_json_str_from_partial(recognized_fn: partial) -> str:
    if not recognized_fn.func in Detector.KnownBlobLikeFn:
        raise DetectorError(f"partial {recognized_fn}.func is not among {Detector.KnownBlobLikeFn}")
    
    mapping = {
        detection.detect_minmax_log: "log",
        detection.detect_minmax_dog: "dog",
        detection.detect_minmax_doh: "doh",
    }
    d = recognized_fn.keywords.copy()
    d["kind"] = mapping[recognized_fn.func]
    return json.dumps(d)

