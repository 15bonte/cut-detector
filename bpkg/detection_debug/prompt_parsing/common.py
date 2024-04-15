from typing import Optional, Any

def get_list(l: list, idx: int, v: Optional[Any] = None) -> Optional[Any]:
    return l[idx] if idx < len(l) else v