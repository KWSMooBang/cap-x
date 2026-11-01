"""Aggregate a bggen batch: python scripts/bggen/aggregate.py OUT/runs/<RUN_TAG> (see capx/bggen/aggregate.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from capx.bggen.aggregate import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
