import pathlib

from decouple import config

API_KEY = None
API_KEY_NAME = "API_KEY"
DATA_PATH = "DATA_PATH"


def get_key_name():
    return API_KEY or config(API_KEY_NAME)


def get_api_key():
    return config(get_key_name())


def get_data_path():
    return config("DATA_PATH", cast=pathlib.Path, default="kaggle/working/")
