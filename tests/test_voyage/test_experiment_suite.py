from voyage import embedding_strategy
from voyage import model
from voyage import experiment_suite
from voyage import experiment


def test_main():
    strategy = embedding_strategy.Truncating()
    embed_model = model.VoyageAiModel.lite()
    experiments = [
                experiment.ExperimentReddit(embed_model, year) for year in range(2012, 2019)
            ]
    suite = experiment_suite.ExperimentSuite(embed_model, strategy, experiments)
    suite.run()
