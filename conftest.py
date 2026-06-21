# pytest runs from the repo root; add it to sys.path so bare imports work.
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
