"""Detector.
a wrapper around factory's SPOT_DETECTION_METHOD
"""

from functools import partial
from typing import Union, Tuple, Callable, Any, Optional, get_args

import numpy as np

from cut_detector.factories.mb_support import detection
from cut_detector.factories.mid_body_detection_factory import MidBodyDetectionFactory
SPOT_DETECTION_METHOD = MidBodyDetectionFactory.SPOT_DETECTION_METHOD

CALLABLE_STR_REPR_PREFIX = "@"

ALLOWED_KW_STR: Tuple[str, ...] = get_args(
    get_args(MidBodyDetectionFactory.SPOT_DETECTION_METHOD)[1]
)

KW_STR_TO_CALLABLE = {
    "cur_log":    detection.cur_log,
    "lapgau":     detection.lapgau,
    "log2_wider": detection.log2_wider,
    "rshift_log": detection.rshift_log,
    "cur_dog":    detection.cur_dog,
    "diffgau":    detection.diffgau,
    "cur_doh":    detection.cur_doh,
    "hessian":    detection.hessian
}

# used to know if the callable is valid and to represent it as a str.
# callable str representation is automatically added a prefix:
# CALLABLE_STR_REPR_PREFIX
# no need to manually add it here.
ALLOWED_CALLABLES_TO_STR: dict[Callable, str] = {
    detection.detect_minmax_log: "mm_log",
    detection.detect_minmax_dog: "mm_dog",
    detection.detect_minmax_doh: "mm_doh",
}
STR_TO_CALLABLE: dict[Callable, str] = {
    v: k
    for k, v in ALLOWED_CALLABLES_TO_STR.items()
}

class DetectorError(Exception):
    pass

class Detector:
    v: SPOT_DETECTION_METHOD

    def __init__(self, v: Union[SPOT_DETECTION_METHOD, str]):
        if isinstance(v, str):
            # str repr or keyword
            if v.startswith(CALLABLE_STR_REPR_PREFIX):
                # str repr
                parts = v.split("|")
                if parts[0][1:] in ALLOWED_CALLABLES_TO_STR.values():
                    self.v = convert_to_callable(v)
                else:
                    raise DetectorError(f"Unknown str repr {parts[0]}")
                
            elif v in ALLOWED_KW_STR:
                # KW
                self.v == v

            else:
                raise DetectorError(f"Invalid str init format: {v}")
            
        elif callable(v):
            # callable
            fn = v.func if isinstance(v, partial) else v
            if fn in ALLOWED_CALLABLES_TO_STR:
                self.v = v
            else:
                raise RuntimeError(f"Unsupported Callable: {v}")
            
        else:
            raise DetectorError(f"Unsupported v (type: {type(v)}): {v}")
        

    def to_factory(self) -> SPOT_DETECTION_METHOD:
        return self.v
    
    def to_str(self) -> str:
        if isinstance(self.v, str):
            return self.v
        elif callable(self.v):
            return convert_to_str(self.v)
        else:
            raise DetectorError(f"Invalid self.v value: {self.v}")

    def try_to_partial(self) -> Optional[partial]:
        if isinstance(self.v, partial):
            return self.v
        elif callable(self.v):
            return partial(self.v)
        elif (p := STR_TO_CALLABLE.get(self.v)) is not None:
            return p
        else:
            return None


def convert_to_callable(s: str) -> Callable[[np.ndarray], np.ndarray]:
    assert s[0] == CALLABLE_STR_REPR_PREFIX, f"(missing prefix) trying to convert str: {s}"
    parts = s.split("|")
    
    func_name = parts[0][1:]
    if (func := STR_TO_CALLABLE.get(func_name)) is None:
        raise DetectorError(f"repr str {s} has unknown func name: {func_name}")
    
    args = parse_args(s[1:])

    return partial(
        func=func,
        **args
    )


def parse_args(args: list[str]) -> dict[str, Union[bool, int, float]]:
    args = {}
    for a in args:
        parts = a.split("=")
        if len(parts) != 2: 
            raise DetectorError(f"ill-formatted arg [{a}] among: {args}")
        arg = parts[0]
        value = parts[1]

        if value == "T":
            value = True
        elif value == "F":
            value = False
        elif value.startswith("i"):
            value = int(value[1:])
        elif value.startswith("f"):
            value = float(value[1:])
        else:
            raise DetectorError(f"Ill-formated value [{value}] among: {args}")
        
        args[arg] = value

    return args


def convert_to_str(f: Callable[[np.ndarray], np.ndarray]) -> str:
    if not callable(f):
        raise DetectorError(f"Invalid non-callable f: {f}")
    
    func = f.func if isinstance(f, partial) else f
    if (func_name := ALLOWED_CALLABLES_TO_STR.get(f)) is None:
        raise DetectorError(f"Unknown callable f: {f}")
    
    # f is a Callable[[np.ndarray], np.ndarray]
    # this means that the only way f has custom arguments is
    # that f is a partial -> we extract the arguments
    # otherwise f has hardcoded value -> no arguments
    if isinstance(f, partial):
        # extraction
        args = [
            f"{k}={convert_arg_to_str(v)}"
            for k, v in f.keywords.items()
        ]

        l = [f"{CALLABLE_STR_REPR_PREFIX}{func_name}"]
        l.extend(args)
        return "|".join(l)
    else:
        # just write with prefix
        return f"{CALLABLE_STR_REPR_PREFIX}{func_name}"
    
def convert_arg_to_str(arg: Union[bool, int, float]) -> str:
    if isinstance(arg, bool):
        return "T" if arg else "F"
    elif isinstance(arg, int):
        return f"i{arg}"
    elif isinstance(arg, float):
        return f"f{arg}"
    else:
        raise DetectorError(f"Ill-formmatted argument {arg} for str conversion")