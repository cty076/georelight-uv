from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from georelight.dataset.schema import validate_dataset
from georelight.dataset.synthetic import SyntheticDatasetConfig, generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a procedural GeoRelight-UV dataset.")
    parser.add_argument("--config", required=True, help="Path to a JSON dataset config.")
    parser.add_argument("--out", required=True, help="Output dataset directory.")
    args = parser.parse_args()

    cfg = SyntheticDatasetConfig.from_json(args.config)
    records = generate_dataset(args.out, cfg)
    counts = validate_dataset(args.out)
    print(f"Generated {len(records)} samples at {args.out}")
    print(f"Splits: {counts}")


if __name__ == "__main__":
    main()
