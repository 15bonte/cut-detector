""" Defines some savers
"""
import sys
from typing import TextIO

class Saver:
    def __init__(self, delay: bool):
        self.delay = delay

    def save(self, data: dict):
        pass

    def delayed_save(self):
        pass

class SaveLogger(Saver):
    """Logs the saved data instead of saving it to the filesystem.

    Note that you can provide a TextIO to redirecto this to other stdio,
    OR to a file.
    """
    def __init__(self, delay: bool = True, file: TextIO = sys.stdout):
        super().__init__(delay)
        self.file = file
    
    def save(self, data: dict):
        print(data, file=self.file)
