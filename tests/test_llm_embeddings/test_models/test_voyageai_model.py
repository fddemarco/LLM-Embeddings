import pytest

from llm_embeddings import settings
from llm_embeddings.models import api_models


@pytest.fixture(name="api_key")
def fixture_api_key():
    return settings.get_config(settings.VOYAGEAI_API_KEY)


def test_01(api_key):
    model = api_models.VoyageAiModel(api_key)
    with pytest.raises(ValueError):
        model.embed([])


def test_02(api_key, snapshot):
    model = api_models.VoyageAiModel(api_key)
    assert snapshot == model.embed(["Sample text"])


def test_03(api_key):
    model = api_models.VoyageAiModel(api_key)
    with pytest.raises(ValueError):
        model.embed(["Sample text {i}" for i in range(0, 128)])
