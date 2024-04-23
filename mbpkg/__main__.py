import sys
from entry import run_binaries

if __name__ == "__main__":
    if len(sys.argv) == 1:
        run_binaries("l")
    elif len(sys.argv) == 2:
        run_binaries(sys.argv[1])
    else:
        print("invalid syntax, see the command list ('python mbpkg l')")

