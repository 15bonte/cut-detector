import numpy as np
import base64
import json

from time import time
from typing import Tuple

def np_array_encode(arr: np.array) -> str:
    """returns a json-and-base64-based representation
    of an nparray.
    see np_array_decode to rebuild the np.array from it
    """
    shape = arr.shape
    dt = arr.dtype.name
    print("arr shape before c order copy:", arr.shape)
    arr = arr.copy(order="C")
    print("arr shape after c order copy:", arr.shape)
    data = base64.b64encode(arr)
    data = data.decode()
    return json.dumps({
        "dtype": dt,
        "shape": shape,
        "data": data
    })

def np_array_decode(data: str) -> np.array:
    """ Returns the np ndarray from the representation
    produced by np_array_encode
    """
    total_start = time()
    start = total_start
    dict = json.loads(data)
    end = time()
    print("json loading took", end-start)

    shape = dict["shape"]
    dt = dict["dtype"]

    start = time()
    raw_bytes = base64.b64decode(dict["data"])
    end = time()
    print("b64 decoding took", end-start)

    start = time()
    flat_arr = np.frombuffer(raw_bytes, dtype=dt)
    end = time()
    total_end = end
    print("np frombuffer took", end-start)
    
    print(f"decoding total time (ms): {total_end-total_start:.2f}")

    return flat_arr.reshape(shape)
