"""Filesystem locations bundled with this experiment."""

from pathlib import Path


def project_root() -> Path:
    """Return the repository root independently of the current directory."""
    return Path(__file__).resolve().parents[1]


def data_path(*parts: str) -> Path:
    """Return a path inside the bundled dataset directory."""
    return project_root().joinpath("data", *parts)


def model_path(name: str) -> Path:
    """Return a path to a bundled pretrained network checkpoint."""
    return project_root() / "DQN" / "exp" / "models2" / name

