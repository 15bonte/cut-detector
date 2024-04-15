from typing import Optional, Union

from .load_source import LoadSource, parse_load_source

PARSE_OUTPUT = Union[LoadSource]

CMD_MAP = {
    "lds": parse_load_source
}

def parse_prompt(p: str) -> Optional[PARSE_OUTPUT]:
    parts = split_prompt(p)
    for idx, p in enumerate(parts):
        print(f"{idx}: {p}")

    if len(parts) == 0:
        return None
    if (fn := CMD_MAP.get(parts[0])) is not None:
        return fn(parts)
    else:
        print(f"Unknown command '{parts[0]}'")
        return None
    

def split_prompt(prompt: str) -> list[str]:
    l = [""]
    idx = 0
    
    # 0: " "
    # 1: "'"
    # 2: '"'
    state = 0

    while idx < len(prompt):
        # " " loop
        if state == 0:
            while idx < len(prompt):
                c = prompt[idx]
                if c == " ":
                    if len(l[len(l)-1]) != 0:  l.append("")
                    idx += 1
                elif c == '"' or c == "'":
                    if len(l[len(l)-1]) != 0: l.append("")
                    idx += 1
                    if c == "'": state   = 1
                    elif c == '"': state = 2
                    break
                else:
                    l[len(l)-1] += c
                    idx += 1
        elif state == 1:
            while idx < len(prompt):
                c = prompt[idx]
                if c == "'":
                    if len(l[len(l)-1]) != 0: l.append("")
                    state = 0
                    idx += 1
                    break
                else:
                    l[len(l)-1] += c
                    idx += 1
        elif state == 2:
            while idx < len(prompt):
                c = prompt[idx]
                if c == '"':
                    if len(l[len(l)-1]) != 0: l.append("")
                    state = 0 
                    idx += 1
                    break
                else:
                    l[len(l)-1] += c
                    idx += 1
        else:
            raise RuntimeError(f"Unknown state {state}")
        
    while len(l[len(l)-1]) == 0:
        l.pop()

    return l