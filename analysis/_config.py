"""Repository paths and optional local analytical inputs."""
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get('AI_NONADOPTION_DATA_DIR', ROOT / 'data' / 'processed')).expanduser().resolve()
SOURCE_DATA = ROOT / 'data'
OUTPUT_DIR = ROOT / 'results'
