import pytest
from voyage import embedding_strategy


@pytest.fixture(name="truncating")
def fixture_truncating():
    return embedding_strategy.Truncating()


@pytest.fixture(name="centroid")
def fixture_centroid():
    return embedding_strategy.Centroid()


class TestStrategy:
    def test_truncating(self, truncating):
        assert str(truncating) == embedding_strategy.TRUNCATING

    def test_centroid(self, centroid):
        assert str(centroid) == embedding_strategy.CENTROID
