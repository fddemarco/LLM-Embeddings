import pytest

from llm_embeddings import settings
from llm_embeddings.models import api_models


@pytest.fixture(name="api_key")
def fixture_api_key():
    return settings.get_config(settings.VOYAGEAI_API_KEY)


@pytest.fixture(name="model")
def fixture_model(api_key):
    return api_models.VoyageAiModel(api_key)


def test_01(model):
    with pytest.raises(ValueError):
        model.embed([])


@pytest.mark.slow
def test_02(model, snapshot):
    assert snapshot == model.embed(["Sample text"])


def test_03(model):
    with pytest.raises(ValueError):
        model.embed(["Sample text {i}" for i in range(0, 129)])


@pytest.mark.slow
def test_04(model, snapshot):
    assert snapshot == model.embed(["Sample text {i}" for i in range(0, 128)])


@pytest.mark.slow
def test_05(model, snapshot):
    assert snapshot == model.embed(["Sample text " * 4096])
