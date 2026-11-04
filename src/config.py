"""Paths. On Colab, data lives on Google Drive; locally it lives in ./data (never committed)."""
import os
from pathlib import Path

IN_COLAB = Path("/content").exists()
_default = (Path("/content/drive/MyDrive/AaharAI/data") if IN_COLAB
            else Path(__file__).resolve().parents[1] / "data")
DATA_DIR = Path(os.environ.get("AAHARAI_DATA", _default))
INDB_DIR = DATA_DIR / "indb"
