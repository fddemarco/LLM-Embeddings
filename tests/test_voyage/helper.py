import os


def setup_env(env_variable, value, f):
    current_path = os.environ[env_variable]
    os.environ[env_variable] = value
    yield f()
    os.environ[env_variable] = current_path


def setup_env_2(env_variable, value, f):
    current_path = os.environ[env_variable]
    os.environ[env_variable] = value
    output = f()
    os.environ[env_variable] = current_path
    return output
