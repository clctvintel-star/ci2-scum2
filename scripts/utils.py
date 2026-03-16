import os
import yaml
from pathlib import Path
from dotenv import load_dotenv


def get_repo_root():
    current = Path(__file__).resolve()

    for parent in [current] + list(current.parents):
        if (parent / "config").exists():
            return parent

    raise RuntimeError("Could not locate repo root (config folder not found)")


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_configs():
    root = get_repo_root()

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
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
    }


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_prompt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
