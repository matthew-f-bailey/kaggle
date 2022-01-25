import pandas as pd
from abc import ABC, abstractmethod

class KNearestNeighbors:

    def __init__(self, data: pd.DataFrame, k: int, distance: str = "euclidian") -> None:
        self._k = k
        distances = self._supported_distances()
        try:
            self._dist_cls: Distance = distances[distance]
        except KeyError:
            raise NotImplementedError(f"Distance of {distance} not supported")

    def _supported_distances(self):
        return {
            "euclidian": Euclidian(),
            "manhattan": Manhattan()
        }

class Distance(ABC):
    """Abstract class for distances"""

    @abstractmethod
    def get_distance(self, X, Y):
        pass

class Euclidian(Distance):

    def get_distance(self, X, Y):
        return 1

class Manhattan(Distance):

    def get_distance(self, X, Y):
        return 2

if __name__ == "__main__":
    knn = KNearestNeighbors(5, distance="manhattan")
    print(knn._dist_cls)