import openai
import tiktoken
import voyageai


class Model:
    def embed(self, sentences):
        self.assert_are_valid(sentences)
        sentences = self.preprocess(sentences)
        sentences = self.truncate(sentences)
        return self.safe_embed(sentences)

    def assert_are_valid(self, sentences):
        self.assert_not_empty(sentences)
        self.assert_allowed_batch_size(sentences)

    def assert_not_empty(self, sentences):
        if not sentences:
            raise ValueError("Input cannot be empty list")

    def assert_allowed_batch_size(self, sentences):
        if len(sentences) > 128:
            raise ValueError(
                f"The batch size limit is 128. Your batch size is {len(sentences)}"
            )

    def safe_embed(self, sentences):
        raise NotImplemented("Should be implemented in a subclass")

    def preprocess(self, sentences):
        return [sentence.replace("\n", " ") for sentence in sentences]

    def truncate(self, sentences):
        raise NotImplemented("Should be implemented in a subclass")


class VoyageAiModel(Model):
    def __init__(self, api_key) -> None:
        self.client = voyageai.Client(api_key=api_key)
        self.n_dimensions = 1024

    def safe_embed(self, sentences):
        response = self.get_response(sentences)
        return response.embeddings

    def get_response(self, sentences):
        return self.client.embed(
            sentences, model="voyage-lite-01", input_type="document", truncation=True
        )

    def truncate(self, sentences):
        return sentences


class Ada2Model(Model):
    def __init__(self, api_key):
        self.client = openai.OpenAI(api_key=api_key)
        self.max_input_tokens = 8192
        self.n_dimensions = 1536
        self.embedding_encoding = "cl100k_base"

    def safe_embed(self, sentences):
        response = self.get_response(sentences)
        return [data.embedding for data in response.data]

    def get_tokens_count(self, sentences):
        response = self.get_response(sentences)
        return [usage for _, usage in response.usage]

    def get_response(self, sentences):
        return self.client.embeddings.create(
            input=sentences, model="text-embedding-ada-002"
        )

    def truncate(self, sentences):
        encoder = tiktoken.get_encoding(self.embedding_encoding)
        return [
            encoder.encode(sentence)[: self.max_input_tokens] for sentence in sentences
        ]
