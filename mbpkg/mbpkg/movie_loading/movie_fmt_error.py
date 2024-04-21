""" Error reported when the format string is not one supported by MovieFmt
"""

class MovieFmtError(Exception):
    invalid_fmt_str: str
    available_fmts: list[str]

    def __init__(self, invalid_fmt_str: str, available_fmts: list[str]):
        super().__init__("Invalid format string '{}'. Expected one of {}".format(
            invalid_fmt_str,
            " ".join(available_fmts),
        ))
        self.invalid_fmt_str = invalid_fmt_str
        self.available_fmts = available_fmts