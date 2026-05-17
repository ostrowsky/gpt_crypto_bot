from __future__ import annotations

import argparse
import json
from pathlib import Path

from v2.sequence_dataset import build_from_jsonl


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "ml_dataset.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build research-only v2 sequence dataset summary")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    result = build_from_jsonl(Path(args.input))
    payload = {
        "input": str(Path(args.input)),
        "summary": dict(result.summary),
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    if args.as_json:
        print(text)
    else:
        summary = payload["summary"]
        print(
            "v2 sequence dataset: "
            f"rows={summary['rows_accepted']}/{summary['rows_read']} "
            f"days={summary['days_covered']} "
            f"symbols={summary['symbols_covered']} "
            f"sequences={summary['sequences_built']} "
            f"transitions={summary['transitions_built']} "
            f"gaps={summary['gap_breaks']} "
            f"status={summary['coverage_status']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

