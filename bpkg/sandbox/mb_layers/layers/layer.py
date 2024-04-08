from abc import ABC, abstractmethod

class BlobLayer(ABC):
    @abstractmethod
    def apply(self, env: dict):
        pass