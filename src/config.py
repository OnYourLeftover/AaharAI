from pathlib import Path

IN_COLAB = Path("/content").exists()
DATA_DIR = (Path("/content/drive/MyDrive/AaharAI/data") if IN_COLAB
            else Path(__file__).resolve().parents[1] / "data")
INDB_DIR = DATA_DIR / "indb"