import tiktoken


MAX_INPUT_TOKENS = 8191
EMBEDDING_ENCODING = "cl100k_base"


class Generator:
    def __init__(self, model, strategy, data):
        self.model = model
        self.strategy = strategy
        self.data = data

    def __repr__(self):
        return f"{self.model}-{self.strategy}"

    def embed(self):
        return self.strategy.embed(self)

    def embed_truncating(self, step=8):
        groups = self.data.groupby("subreddit")["text"].apply(" ".join).reset_index()
        texts = self.truncate_texts(groups["text"].to_list())
        
        embeddings = []
        for i in range(0, len(texts), step):
            embeddings += self.model.embed(texts[i : i + step])

        return {"subreddits": groups["subreddit"].tolist(), "embeddings": embeddings}
    
    def truncate_texts(self, texts):
        encoder = tiktoken.get_encoding(EMBEDDING_ENCODING)
        return [encoder.encode(text.replace("\n", " "))[:MAX_INPUT_TOKENS] for text in texts]

    def embed_centroid(self):
        return {"subreddits": ["a", "b", "c"], "embeddings": [[0.5], [-1.1], [0.1]]}
