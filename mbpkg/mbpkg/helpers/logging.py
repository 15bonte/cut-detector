import contextlib
from typing import Optional, TextIO

class Logger:
    """ Do not instantiate this directly, use `manage_logger` instead (and see its doc)
    """
    file: TextIO
    should_print: bool

    def __init__(self, filepath: Optional[str] = None, should_print: bool = True):
        if isinstance(filepath, str):
            self.file = open(filepath, "w")
        else:
            self.file = None
        self.should_print = should_print

    def log(self, *args, sep: str = " ", end: str = "\n"):
        if self.should_print:
            print(*args, sep=sep, end=end)
        if self.file is not None:
            print(*args, sep=sep, end=end, file=self.file)

    def close(self):
        if self.file is not None:
            self.file.close()


@contextlib.contextmanager
def manage_logger(path: Optional[str] = None, should_print: bool = True):
    """ A context manager for the logger.
    You can use it that way:
    ```python
    with manager_logger("path/to/log/file.txt", True) as l:
        l.print("hello world")
    ```
    Don't hesitate to pass this instance in subfonctions that would take
    a `Logger
    """
    try:
        logger = Logger(path, should_print)
        yield logger
    finally:
        logger.close()


