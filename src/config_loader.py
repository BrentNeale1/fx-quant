# src/config_loader.py
"""
Loads config/system.yaml and config/.env for the fx-quant project.
"""

from pathlib import Path
from dotenv import load_dotenv
import yaml


def get_project_root() -> Path:
    """Return the fx-quant project root (parent of src/)."""
    return Path(__file__).resolve().parents[1]


def load_config(path=None) -> dict:
    """
    Load config/system.yaml and config/.env.

    Parameters
    ----------
    path : str or Path, optional
        Explicit path to a YAML config file.  Defaults to
        <project_root>/config/system.yaml.

    Returns
    -------
    dict
        Parsed YAML contents.
    """
    root = get_project_root()

    # Always load .env as a side effect
    env_path = root / "config" / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    if path is None:
        path = root / "config" / "system.yaml"

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    return cfg
