import voyageai


class VoyageAiModel:
    def __init__(self, api_key) -> None:
        self.client = voyageai.Client(api_key=api_key)

    def embed(self, texts):
        if not texts:
            raise ValueError("Input cannot be empty list")

        if len(texts) > 128:
            raise ValueError(
                f"The batch size limit is 128. Your batch size is {len(texts)}"
            )

        return self.client.embed(
            texts, model="voyage-lite-01", input_type="document"
        ).embeddings
