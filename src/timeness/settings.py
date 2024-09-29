from decouple import config


OPENAI_API_KEY = "OPENAI_API_KEY"
VOYAGEAI_API_KEY = "VOYAGEAI_API_KEY"


def get_config(env_var):
    return config(env_var)
