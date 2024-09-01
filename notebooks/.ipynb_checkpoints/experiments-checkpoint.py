import polars as pl
import numpy as np
import constants


def mean_experiment(
    k,
    party,
    model,
    embeddings,
    by=constants.DATE_COL,
    dimension_col=constants.DIMENSION_COL,
    experiment_col=constants.EXPERIMENT_COL,
    k_col=constants.K_COL,
    party_col=constants.PARTY_COL,
    model_col=constants.MODEL_COL,
):
    schema = Schema(by, dimension_col, experiment_col, k_col, party_col, model_col)
    return Experiment(k, party, model, embeddings, schema).get_timeness()


class Schema:
    def __init__(
        self, index, dimension_col, experiment_col, k_col, party_col, model_col
    ):
        self.index_col = index
        self.dimension_col = dimension_col
        self.experiment_col = experiment_col
        self.k_col = k_col
        self.party_col = party_col
        self.model_col = model_col

    def get_cols(self):
        return [getattr(self, x) for x in dir(self) if x.endswith("_col")]

    def set_index(self, embeddings):
        return (
            embeddings.to_pandas().set_index(self.index_col).sort_index(ascending=True)
        )

    def set_dimension(self, embeddings, dimension):
        embeddings[self.dimension_col] = dimension

    def set_experiment(self, embeddings, experiment):
        embeddings[self.experiment_col] = experiment

    def set_k(self, embeddings, k):
        embeddings[self.k_col] = k

    def set_party(self, embeddings, party):
        embeddings[self.party_col] = party

    def set_model(self, embeddings, model):
        embeddings[self.model_col] = model

    def reset_index(self, embeddings):
        return embeddings.reset_index()[self.get_cols()]


class Experiment:
    def __init__(self, k, party, model, embeddings, schema):
        self.k = k
        self.party = party
        self.model = model
        self.schema = schema
        self.experiment = f"{k}-{party}-{model}"
        self.embeddings = self.schema.set_index(embeddings)

    def get_timeness(self):
        embeddings = self.embeddings.copy()
        self.schema.set_dimension(embeddings, self.get_dimension())

        self.schema.set_experiment(embeddings, self.experiment)
        self.schema.set_k(embeddings, self.k)
        self.schema.set_party(embeddings, self.party)
        self.schema.set_model(embeddings, self.model)

        return self.schema.reset_index(embeddings)

    def get_dimension(self):
        direction = self.get_direction()
        return self.embeddings.dot(direction)

    def get_direction(self):
        return self.get_top() - self.get_bottom()

    def get_bottom(self):
        return self.get_extremes(True)

    def get_top(self):
        return self.get_extremes(False)

    def get_extremes(self, smallest):
        if smallest:
            extremes = self.embeddings.iloc[: self.k]
        else:
            extremes = self.embeddings.iloc[-self.k :]
        return extremes.mean()
