import pytest

from voyage import model


@pytest.fixture(
    name="model_config",
    params=[model.VoyageAiConfig.LITE.value, model.VoyageAiConfig.BASE.value],
)
def fixture_model_config(request):
    return request.param


@pytest.fixture(name="voyage_model")
def fixture_voyage_model(model_config):
    return model.VoyageAiModel(model_config)


def test_name(voyage_model, model_config):
    n_tokens = model_config[model.N_TOKENS]
    model_name = model_config[model.NAME]
    assert str(voyage_model) == f"{n_tokens}-{model_name}"


@pytest.mark.slow
def test_embed(voyage_model, model_config):
    n_dimensions = model_config[model.N_DIMENSIONS]
    output = voyage_model.embed(["Example str"])[0]
    assert len(output) == n_dimensions


def test_lite():
    voyage_lite = model.VoyageAiModel.lite()
    assert voyage_lite.name == model.VoyageAiConfig.LITE.value[model.NAME]
    assert voyage_lite.n_tokens == model.VoyageAiConfig.LITE.value[model.N_TOKENS]


def test_base():
    voyage_base = model.VoyageAiModel.base()
    assert voyage_base.name == model.VoyageAiConfig.BASE.value[model.NAME]
    assert voyage_base.n_tokens == model.VoyageAiConfig.BASE.value[model.N_TOKENS]
