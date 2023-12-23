import pathlib

from dotenv import load_dotenv
from decouple import config

API_KEY = "API_KEY"
DATA_PATH = "DATA_PATH"


def get_api_key():
    return config(API_KEY)


def get_data_path():
    return config("DATA_PATH", cast=pathlib.Path, default="kaggle/working/")


load_dotenv()
