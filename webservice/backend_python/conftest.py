# Ensure project root is importable for tests
import os
import sys

# Add the directory containing main.py to sys.path when running pytest
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
