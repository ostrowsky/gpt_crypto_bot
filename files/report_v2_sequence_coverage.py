from __future__ import annotations

import argparse
import json
from pathlib import Path

from v2.coverage import build_coverage_audit
from v2.sequence_dataset import build_from_jsonl


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "ml_dataset.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description="Report research-only v2 sequence coverage")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    dataset = build_from_jsonl(Path(args.input))
    audit = build_coverage_audit(dataset)
    payload = {
        "input": str(Path(args.input)),
        "summary": dict(audit.summary),
        "by_day": audit.by_day,
        "by_timeframe": audit.by_timeframe,
        "fragmented_slices": list(audit.fragmented_slices),
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    if args.as_json:
        print(text)
    else:
        s = payload["summary"]
        print(
            "v2 coverage: "
            f"status={s['coverage_status']} "
            f"days={s['days_covered']} "
            f"transitions={s['transitions_built']} "
            f"density={s['transition_density']:.4f} "
            f"longest={s['longest_sequence_bars']} bars/"
            f"{s['longest_sequence_minutes']} min"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

