import pathlib
import pickle

import pandas as pd


class EmbeddingsLoader:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

    def load_embeddings(self):
        with open(self.pathname(), "rb") as file:
            stored_data = pickle.load(file)
            subreddits = stored_data["subreddits"]
            embeddings = stored_data["embeddings"]
        return pd.DataFrame(
            embeddings, index=subreddits, columns=range(len(embeddings[0]))
        )

    def pathname(self):
        return pathlib.Path("embeddings") / "pickles" / self.filename()

    def filename(self, extension="pkl"):
        return f"{self.model.filename(self)}.{extension}"

    def filename_voyage(self):
        return self.dataset.filename_voyage(self)

    def filename_voyage_speeches(self):
        return "embeddings-sotu-voyage-lite-4096"

    def filename_voyage_literals(self):
        return "embeddings-years-literals-voyage-lite"

    def filename_ada(self):
        return self.dataset.filename_ada(self)

    def filename_ada_speeches(self):
        return "embeddings-sotu-ada-002-8192"

    def filename_ada_literals(self):
        return "embeddings-years-literals-ada"


class Embedder:
    def filename(self, loader):
        raise NotImplemented


class VoyageEmbedder(Embedder):
    def filename(self, loader):
        return loader.filename_voyage()


class AdaEmbedder(Embedder):
    def filename(self, loader):
        return loader.filename_ada()


class DatasetFilename:
    def filename_voyage(self, loader):
        raise NotImplemented

    def filename_ada(self, loader):
        raise NotImplemented


class SpeechesFilename(DatasetFilename):
    def filename_voyage(self, loader):
        return loader.filename_voyage_speeches()

    def filename_ada(self, loader):
        return loader.filename_ada_speeches()


class LiteralsFilename(DatasetFilename):
    def filename_voyage(self, loader):
        return loader.filename_voyage_literals()

    def filename_ada(self, loader):
        return loader.filename_ada_literals()
