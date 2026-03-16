import os
import yaml
from pathlib import Path
from dotenv import load_dotenv


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_configs():
    root = Path(__file__).resolve().parents[1]

    firms = load_yaml(root / "config" / "firms.yaml")
    paths = load_yaml(root / "config" / "paths.yaml")
    settings = load_yaml(root / "config" / "settings.yaml")

    return firms, paths, settings


def load_env(env_path):

    load_dotenv(env_path)

    return {
        "SERPAPI_API_KEY": os.getenv("SERPAPI_API_KEY"),
        "REDDIT_CLIENT_ID": os.getenv("REDDIT_CLIENT_ID"),
        "REDDIT_SECRET": os.getenv("REDDIT_SECRET"),
        "REDDIT_AGENT": os.getenv("REDDIT_AGENT"),
    }


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
