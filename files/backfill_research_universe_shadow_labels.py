from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from research_universe_shadow_collector import DATASET_FILE, backfill_mature_labels


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill all mature labels in the research-universe shadow dataset."
    )
    parser.add_argument("--dataset", type=Path, default=DATASET_FILE)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    result = asyncio.run(
        backfill_mature_labels(
            dataset_file=args.dataset,
            concurrency=max(1, args.concurrency),
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not result.get("pairs_failed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
