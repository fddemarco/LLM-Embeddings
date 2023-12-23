import pytest
import pandas as pd
import pathlib
import pickle

from voyage import experiment
from voyage import settings
from test_voyage import helper


@pytest.fixture(name="experiment_name")
def fixture_experiment_name():
    return "experiment"


@pytest.fixture(name="experiment_year")
def fixture_experiment_year():
    return 2012


@pytest.fixture(name="embeddings")
def fixture_embeddings():
    return [[0.1], [-1.5], [2.0]]


@pytest.fixture(name="data_path")
def fixture_data_path():
    return "data"


@pytest.fixture(name="voyage_experiment")
def fixture_voyage_experiment(experiment_name, experiment_year, data_path):
    def f():
        return experiment.ExperimentReddit(experiment_name, experiment_year)

    yield from helper.setup_env(settings.DATA_PATH, data_path, f)


def test_experiment_filename(
    experiment_name, experiment_year, voyage_experiment, data_path
):
    expected = (
        pathlib.Path(data_path)
        / f"embeddings/embeddings-{experiment_year}-{experiment_name}.pkl"
    )
    assert expected == voyage_experiment.embeddings_filename()


def test_save_embeddings(voyage_experiment, embeddings):
    voyage_experiment.save_embeddings(embeddings)
    with open(voyage_experiment.embeddings_filename(), "rb") as f:
        saved_embeddings = pickle.load(f)
    assert embeddings == saved_embeddings


def test_load_data(voyage_experiment, tmp_path):
    data_path = tmp_path / "data"
    data_path.mkdir()
    filename = data_path / voyage_experiment.data_filename()

    df = pd.DataFrame(
        {experiment.SUBREDDIT_COL: ["a", "b"], experiment.TEXT_COL: ["hi", "bye"]}
    )
    df.to_parquet(filename)

    def f():
        return voyage_experiment.load_data()

    read_data = helper.setup_env_2(settings.DATA_PATH, str(data_path), f)
    pd.testing.assert_frame_equal(df, read_data)
