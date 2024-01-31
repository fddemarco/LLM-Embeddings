import polars as pl
import pytest

from llm_embeddings import settings
from llm_embeddings.models import api_models
from llm_embeddings.models import datasets


@pytest.fixture(name="schema")
def fixture_schema():
    return datasets.Schema("text", "group")


@pytest.fixture(name="data")
def fixture_data(schema):
    return pl.DataFrame(
        {schema.text_key: ["Sample text 1", "Sample text 2"], schema.group_key: [0, 1]}
    )


@pytest.fixture(name="dataset")
def fixture_dataset(data, schema):
    return datasets.Dataset(data, schema)


@pytest.fixture(name="api_key")
def fixture_api_key():
    return settings.get_config(settings.VOYAGEAI_API_KEY)


@pytest.fixture(name="model")
def fixture_model(api_key):
    return api_models.VoyageAiModel(api_key)


@pytest.fixture(name="openai_api_key")
def fixture_api_key():
    return settings.get_config(settings.OPENAI_API_KEY)


@pytest.fixture(name="ada_model")
def fixture_model(openai_api_key):
    return api_models.Ada2Model(openai_api_key)


@pytest.mark.slow
def test_01(snapshot, dataset, model):
    assert snapshot == dataset.embed_text(model)


def test_02(snapshot, dataset, ada_model):
    assert snapshot == dataset.embed_text(ada_model)
