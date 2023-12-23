import pickle

import pandas as pd

from voyage import settings


DATA_FILENAME = "%s_truncated_data.parquet"
SUBREDDIT_COL = "subreddit"
TEXT_COL = "text"


class Experiment:
    def load_data(self):
        return pd.read_parquet(self.data_filepath(), columns=[SUBREDDIT_COL, TEXT_COL])

    def data_filepath(self):
        return settings.get_data_path() / self.data_filename()

 
    def save_embeddings(self, embeddings):
        filename = self.embeddings_filename()
        with open(filename, "wb") as f_out:
            pickle.dump(embeddings, f_out, protocol=pickle.HIGHEST_PROTOCOL)


class ExperimentReddit(Experiment):
    def __init__(self, name, year):
        self.name = name
        self.year = year

    def data_filename(self):
       return DATA_FILENAME % self.year
 
    def embeddings_filename(self):
        folder = settings.get_data_path() / "embeddings"
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"embeddings-{self.year}-{self.name}.pkl"


class ExperimentSpeeches(Experiment):
    def data_filename(self):
        return "speech_dataset.parquet"
 
    def embeddings_filename(self):
        folder = settings.get_data_path() / "speeches"
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"embeddings-sotu.pkl"
