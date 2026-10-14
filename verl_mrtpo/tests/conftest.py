import sys
from pathlib import Path

# Run the tests from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
