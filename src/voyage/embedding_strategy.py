import abc

TRUNCATING = "truncating"
CENTROID = "centroid"


class Strategy(metaclass=abc.ABCMeta):
    def __str__(self):
        return self.value

    @abc.abstractmethod
    def embed(self, generator):
        raise NotImplementedError


class Truncating(Strategy):
    def __init__(self):
        self.value = TRUNCATING

    def embed(self, generator):
        return generator.embed_truncating()


class Centroid(Strategy):
    def __init__(self):
        self.value = CENTROID

    def embed(self, generator):
        return generator.embed_centroid()
