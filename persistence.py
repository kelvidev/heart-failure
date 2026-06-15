import os
import pickle

import config


def ensure_models_dir():
    os.makedirs(config.MODELS_DIR, exist_ok=True)


def save_pickle(obj, path):
    ensure_models_dir()
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)
