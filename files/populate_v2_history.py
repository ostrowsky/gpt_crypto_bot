from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import config
from v2.history_population import build_requests, populate_requests
from v2.history_store import LocalHistoryStore


ROOT = Path(__file__).resolve().parent
DEFAULT_STORE = ROOT / ".runtime" / "v2_history"


def main() -> int:
    parser = argparse.ArgumentParser(description="Populate research-only v2 canonical history store")
    parser.add_argument("--symbols", nargs="*")
    parser.add_argument("--timeframes", nargs="*", default=["15m", "1h"])
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--store-root", default=str(DEFAULT_STORE))
    parser.add_argument("--output")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    symbols = args.symbols or config.load_watchlist()
    requests = build_requests(symbols, args.timeframes, days=args.days)
    report = asyncio.run(
        populate_requests(
            LocalHistoryStore(Path(args.store_root)),
            requests,
            target_days=args.days,
        )
    )
    payload = report.to_dict()
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    if args.as_json:
        print(text)
    else:
        print(
            "v2 history population: "
            f"requests={len(payload['requested_rows'])} "
            f"valid_symbols={len(payload['valid_symbols'])} "
            f"ratio={payload['valid_symbol_ratio']:.2%} "
            f"passed={payload['coverage_passed']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

