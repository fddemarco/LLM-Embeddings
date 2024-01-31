import openai
import tiktoken
import voyageai


class Model:
    def embed(self, sentences):
        self.assert_sentences(sentences)
        sentences = self.truncate_sentences(sentences)
        return self.safe_embed(sentences)

    def assert_sentences(self, sentences):
        self.assert_not_empty(sentences)
        self.assert_batch_size(sentences)

    def assert_not_empty(self, sentences):
        if not sentences:
            raise ValueError("Input cannot be empty list")

    def assert_batch_size(self, sentences):
        if len(sentences) > 128:
            raise ValueError(
                f"The batch size limit is 128. Your batch size is {len(sentences)}"
            )

    def safe_embed(self, sentences):
        raise NotImplemented("Should be implemented in a subclass")

    def truncate_sentences(self, sentences):
        raise NotImplemented("Should be implemented in a subclass")


class VoyageAiModel(Model):
    def __init__(self, api_key) -> None:
        self.client = voyageai.Client(api_key=api_key)

    def safe_embed(self, sentences):
        return self.client.embed(
            sentences, model="voyage-lite-01", input_type="document", truncation=True
        ).embeddings

    def truncate_sentences(self, sentences):
        return sentences


class Ada2Model(Model):
    def __init__(self, api_key):
        self.client = openai.OpenAI(api_key=api_key)
        self.max_input_tokens = 8192
        self.embedding_encoding = "cl100k_base"

    def safe_embed(self, sentences):
        response = self.client.embeddings.create(
            input=sentences, model="text-embedding-ada-002"
        )
        return [data.embedding for data in response.data]

    def truncate_sentences(self, sentences):
        encoder = tiktoken.get_encoding(self.embedding_encoding)
        return [
            encoder.encode(sentence)[: self.max_input_tokens] for sentence in sentences
        ]
