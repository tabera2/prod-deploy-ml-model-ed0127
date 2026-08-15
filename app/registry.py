"""A tiny file-based model registry: versioned artifacts + an active pointer.

Layout on disk:
    models/
      v1/model.joblib
      v2/model.joblib
      ACTIVE        <- a text file holding the active version, e.g. "v2"
"""
import joblib
from pathlib import Path

MODELS_DIR = Path("models")
ACTIVE_FILE = MODELS_DIR / "ACTIVE"


def active_version() -> str:
    # The single source of truth for which version is live.
    return ACTIVE_FILE.read_text().strip()


def load_model(version: str):
    path = MODELS_DIR / version / "model.joblib"
    if not path.exists():
        raise FileNotFoundError(f"no artifact for version {version!r}")
    return joblib.load(path)


def set_active(version: str) -> None:
    # Flip the live version by rewriting one pointer file — instant rollback.
    if not (MODELS_DIR / version / "model.joblib").exists():
        raise FileNotFoundError(f"cannot activate missing version {version!r}")
    ACTIVE_FILE.write_text(version)
