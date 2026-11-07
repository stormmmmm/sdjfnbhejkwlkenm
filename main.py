"""Запуск: python app/main.py или python -m app.main из корня проекта."""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gui import main

if __name__ == "__main__":
    main()
