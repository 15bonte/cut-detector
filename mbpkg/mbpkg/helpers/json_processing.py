from typing import Optional, Any, Union

class ExtractionError(Exception):
    pass

def extract(
        data: dict, 
        path: str, 
        kind: Optional[Union[type, list[type]]] = None,
        allow_missing: bool = False
        ) -> Any:
    
    parts = path.split("/")
    cur_data = data

    for p in parts:
        if isinstance(cur_data, dict):
            new_cur_data = cur_data.get(p)
            if new_cur_data is None:
                if allow_missing:
                    return None
                else:
                    raise ExtractionError(f"invalid path '{path}': key '{p}' not found in\n{cur_data}")
            cur_data = new_cur_data
        else:
            raise ExtractionError(
                f"invalid path '{path}': at key '{p}' found {cur_data} instead of dict"
            )
        
    if kind is None:
        return cur_data
    
    if isinstance(kind, type):
        types = [kind]
    elif isinstance(kind, list):
        types = kind
    else:
        raise ExtractionError(f"Invalid kind field: {kind}; expected type/list[type]")
    

    if isinstance(cur_data, tuple(types)):
        return cur_data
    else:
        raise ExtractionError(f"Data {cur_data} is not among {tuple(types)}")
    


def assert_extraction(value: Any, kind: Union[type, list[type]]) -> Any:
    if isinstance(kind, type):
        types = [kind]
    else:
        types = kind
    
    if isinstance(value, tuple(types)):
        return value
    else:
        raise ExtractionError(f"value {value} not among {tuple(types)}")

