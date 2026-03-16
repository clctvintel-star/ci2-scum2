import os
import yaml
from dotenv import load_dotenv

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_env(env_path):
    load_dotenv(env_path)
    return {
        "SERPAPI_API_KEY": os.getenv("SERPAPI_API_KEY"),
        "REDDIT_CLIENT_ID": os.getenv("REDDIT_CLIENT_ID"),
        "REDDIT_SECRET": os.getenv("REDDIT_SECRET"),
        "REDDIT_AGENT": os.getenv("REDDIT_AGENT"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    }

def ensure_dirs(paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def load_prompt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
