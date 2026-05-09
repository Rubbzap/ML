from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"

DEFAULT_TICKER = "AAPL"
DEFAULT_START = "2015-01-01"
DEFAULT_END = None
RANDOM_STATE = 42
