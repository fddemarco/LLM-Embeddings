import pytest

from llm_embeddings import settings
from llm_embeddings.models import api_models


@pytest.fixture(name="api_key")
def fixture_api_key():
    return settings.get_config(settings.OPENAI_API_KEY)


@pytest.fixture(name="model")
def fixture_model(api_key):
    return api_models.Ada2Model(api_key)


@pytest.mark.slow
def test_02(model, snapshot):
    assert snapshot == model.embed(["Sample text"])


@pytest.mark.slow
def test_03(model, snapshot):
    assert snapshot == model.embed(["Sample text" * 8192])
