from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from georelight.dataset.ambientcg import AmbientCGConfig, generate_from_ambientcg
from georelight.dataset.schema import validate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a real-material GeoRelight-UV dataset from ambientCG.")
    parser.add_argument("--config", required=True, help="Path to ambientCG dataset config JSON.")
    parser.add_argument("--out", required=True, help="Output dataset directory.")
    parser.add_argument("--raw", required=True, help="Raw download/extraction cache directory.")
    args = parser.parse_args()

    cfg = AmbientCGConfig.from_json(args.config)
    records = generate_from_ambientcg(args.out, args.raw, cfg)
    counts = validate_dataset(args.out)
    print(f"Generated {len(records)} real-material samples at {args.out}")
    print(f"Splits: {counts}")


if __name__ == "__main__":
    main()
