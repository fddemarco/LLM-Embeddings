import openai
import voyageai


class Model:
    def embed(self, texts):
        self.assert_texts(texts)
        return self.safe_embed(texts)

    def safe_embed(self, texts):
        raise NotImplemented("Should be implemented in a subclass")

    def assert_texts(self, texts):
        self.assert_not_empty(texts)
        self.assert_batch_size(texts)

    def assert_not_empty(self, texts):
        if not texts:
            raise ValueError("Input cannot be empty list")

    def assert_batch_size(self, texts):
        if len(texts) > 128:
            raise ValueError(
                f"The batch size limit is 128. Your batch size is {len(texts)}"
            )


class VoyageAiModel(Model):
    def __init__(self, api_key) -> None:
        self.client = voyageai.Client(api_key=api_key)

    def safe_embed(self, texts):
        return self.client.embed(
            texts, model="voyage-lite-01", input_type="document"
        ).embeddings


class Ada2Model(Model):
    def __init__(self, api_key):
        self.client = openai.OpenAI(api_key=api_key)

    def safe_embed(self, sentences):
        response = self.client.embeddings.create(
            input=sentences, model="text-embedding-ada-002"
        )
        return [data.embedding for data in response.data]
