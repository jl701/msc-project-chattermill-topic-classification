"""Start the loopback-only Chattermill inference API."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from msc_project.serving.server import main

if __name__ == '__main__':
    raise SystemExit(main())
