import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def load_config(path: str = "configs/config.yaml") -> dict[str, Any]:
    """Load YAML config from a local path."""
    load_dotenv()
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")

    return config


def get_env_or_config(name: str, config_value: Any, default: Any = None) -> Any:
    """Return environment value first, then config value, then default."""
    env_value = os.getenv(name)
    if env_value not in (None, ""):
        return env_value
    if config_value is not None:
        return config_value
    return default
