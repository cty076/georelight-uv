import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from georelight.baselines.evaluate_heuristics import main


if __name__ == "__main__":
    main()
