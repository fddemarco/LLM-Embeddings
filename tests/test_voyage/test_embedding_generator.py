import pytest
import pandas as pd

from voyage import embedding_generator
from voyage import embedding_strategy
from voyage import model


@pytest.fixture(
    name="strategy",
    params=[embedding_strategy.Truncating(), embedding_strategy.Centroid()],
)
def fixture_strategy(request):
    return request.param


@pytest.fixture(name="experiment_model")
def fixture_experiment_model():
    return model.VoyageAiModel.lite()


@pytest.fixture(name="data")
def fixture_data():
    return pd.DataFrame(
        {
            "subreddit": ["a", "b", "c", "a", "c", "a"],
            "text": ["text 1", "text 2", "3 text", "example", "hello", "world!"],
        }
    )


@pytest.fixture(name="generator")
def fixture_generator(experiment_model, strategy, data):
    return embedding_generator.Generator(experiment_model, strategy, data)


def test_name(generator, strategy, experiment_model):
    assert f"{generator}" == f"{experiment_model}-{strategy}"


@pytest.mark.slow
def test_embed(generator):
    embeddings = generator.embed()
    assert embeddings["subreddits"] == ["a", "b", "c"]
    assert len(embeddings["embeddings"]) == 3
