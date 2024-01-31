import voyageai


class VoyageAiModel:
    def __init__(self, api_key) -> None:
        self.client = voyageai.Client(api_key=api_key)

    def embed(self, texts):
        if texts:
            return self.client.embed(
                texts, model="voyage-lite-01", input_type="document"
            ).embeddings
        raise ValueError("Input cannot be empty list")
