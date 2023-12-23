import time


class Generator:
    def __init__(self, model, strategy, data):
        self.model = model
        self.strategy = strategy
        self.data = data

    def __repr__(self):
        return f"{self.model}-{self.strategy}"

    def embed(self):
        return self.strategy.embed(self)

    def embed_truncating(self):
        groups = self.data.groupby("subreddit")["text"].apply(" ".join).reset_index()
        texts = groups.text.to_list()
        embeddings = []

        for i in range(0, len(texts), 8):
            embeddings += self.model.embed(texts[i : i + 8])

        return {"subreddits": groups.subreddit.tolist(), "embeddings": embeddings}

    def embed_centroid(self):
        return {"subreddits": ["a", "b", "c"], "embeddings": [[0.5], [-1.1], [0.1]]}
