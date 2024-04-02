
SANDBOX_PROJECT_LIST = """
=== Sandbox Project List ===
- hw: Hello World proof-of-concept
"""

SANDBOX_HELP = """
=== Sandbox Project List ===
Sandbox projects are an intermediate between a library and
a binary.
They are neither meant to be controlled through API like libraries,
nor to be "simple executables" without configurations (like the
library runners).

They are meant to be modified directly.
"""
__doc__ = SANDBOX_HELP

def run_sandbox_project(name: str):
    if name == "list":
        print(SANDBOX_PROJECT_LIST)
    elif name == "help":
        print(SANDBOX_HELP)
    elif name == "hw":
        from .hw import hello_world
        hello_world()