from llm_embeddings.models import api_models


def test_01():
    model = api_models.VoyageAiModel()
    assert not model.embed([])
