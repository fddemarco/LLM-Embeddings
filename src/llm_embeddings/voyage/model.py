import time
import abc
import enum

import openai
import voyageai

from llm_embeddings import settings


API_KEY = settings.get_api_key()
NAME = "name"
N_TOKENS = "n_tokens"
N_DIMENSIONS = "n_dimensions"


class Model(metaclass=abc.ABCMeta):
    def __init__(self, n_tokens, name):
        self.n_tokens = n_tokens
        self.name = name

    def __repr__(self):
        return f"{self.n_tokens}-{self.name}"

    @abc.abstractmethod
    def embed(self, sentences):
        raise NotImplementedError


class VoyageAiModel(Model):
    @classmethod
    def base(cls):
        return cls(VoyageAiConfig.BASE.value)

    @classmethod
    def lite(cls):
        return cls(VoyageAiConfig.LITE.value)

    def __init__(self, model_config):
        n_tokens = model_config[N_TOKENS]
        name = model_config[NAME]
        super().__init__(n_tokens, name)
        voyageai.api_key = API_KEY

    def embed(self, sentences):
        output = voyageai.get_embeddings(sentences, self.name)
        time.sleep(1)
        return output


class Ada2Model(Model):
    def __init__(self):
        self.client = openai.OpenAI(api_key=API_KEY)
        super().__init__(8192, "ada-002")

    def embed(self, sentences):
        response = self.client.embeddings.create(
            input=sentences, model="text-embedding-ada-002"
        )
        return [data.embedding for data in response.data]


class VoyageAiConfig(enum.Enum):
    BASE = {NAME: "voyage-base-01", N_TOKENS: 4096, N_DIMENSIONS: 1024}
    LITE = {NAME: "voyage-lite-01", N_TOKENS: 4096, N_DIMENSIONS: 1024}
