from typing import Literal, Optional

from .prompt_parsing import parse_prompt

APP_MODE = Literal["cli"]


def run_app(mode: Optional[APP_MODE] = "cli"):
    if mode is None or mode == "cli":
        run_cli_app()
    else:
        raise RuntimeError(f"Unknown mode {mode}")


def run_cli_app():
    print("=== Detection debugging CLI App ===")
    print("type 'help' to view all commands")
    print("type 'exit' to leave the app")
    print("")
    while True:
        ipt = input(">> ")
        if ipt == "exit" or ipt == "exit()":
            break
        
        print("ipt>:", ipt)
        print(parse_prompt(ipt))