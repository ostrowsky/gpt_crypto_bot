from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import watchlist_top_gainer_model


def _make_record(sym: str, ts: str, *, tf: str = "15m", positive: bool = False) -> dict:
    boost = 1.0 if positive else -1.0
    return {
        "id": f"{sym}_{tf}_{ts.replace(':', '').replace('-', '')}",
        "sym": sym,
        "tf": tf,
        "ts_signal": ts,
        "bar_ts": int(datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() * 1000),
        "signal_type": "trend" if positive else "none",
        "is_bull_day": bool(positive),
        "hour_utc": datetime.fromisoformat(ts.replace("Z", "+00:00")).hour,
        "dow": datetime.fromisoformat(ts.replace("Z", "+00:00")).weekday(),
        "f": {
            "adx": 40.0 + 10.0 * boost,
            "atr_pct": 1.5 + 0.1 * boost,
            "body_pct": 1.2 + 0.2 * boost,
            "btc_momentum_4h": 0.8 + 0.2 * boost,
            "btc_vs_ema50": 1.0 + 0.2 * boost,
            "close_vs_ema20": 3.0 + 1.0 * boost,
            "close_vs_ema200": 5.0 + 1.0 * boost,
            "close_vs_ema50": 4.0 + 1.0 * boost,
            "daily_range": 7.0 + 0.5 * boost,
            "ema20_vs_ema50": 2.0 + 0.3 * boost,
            "ema50_vs_ema200": 1.0 + 0.2 * boost,
            "lower_wick_pct": 0.3,
            "macd_hist_norm": 0.2 * boost,
            "market_vol_24h": 10.0 + boost,
            "rsi": 62.0 + 6.0 * boost,
            "slope": 0.8 + 0.3 * boost,
            "upper_wick_pct": 0.2,
            "vol_x": 1.8 + 0.3 * boost,
        },
        "seq": [[1.0, 1.01, 0.99, 1.0, 1.2 + 0.1 * boost, 0.4 + 0.1 * boost, 20.0 + 5.0 * boost, 55.0 + 5.0 * boost, 0.1 * boost, 1.0] for _ in range(20)],
        "seq_feature_names": ["close_norm", "high_norm", "low_norm", "open_norm", "vol_x", "slope", "adx", "rsi", "macd_hist_norm", "atr_pct"],
        "labels": {},
    }


class TestWatchlistTopGainerModel(unittest.TestCase):
    def test_backfill_from_local_reports_annotates_ml_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset = tmp / "ml_dataset.jsonl"
            reports = tmp / "reports"
            reports.mkdir(parents=True, exist_ok=True)

            rows = [
                _make_record("AAAUSDT", "2026-04-01T08:00:00Z", positive=True),
                _make_record("BBBUSDT", "2026-04-01T08:00:00Z", positive=False),
            ]
            dataset.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")

            report = {
                "target_day_local": "2026-04-01",
                "phase": "final",
                "settings": {"timezone": "Europe/Budapest"},
                "watchlist_top_gainers": [
                    {"symbol": "AAAUSDT", "status": "bought", "day_change_pct": 25.0, "quote_volume_24h": 1000000.0}
                ],
            }
            (reports / "top_gainer_critic_2026-04-01_final.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            old_watchlist = watchlist_top_gainer_model.top_gainer_critic.WATCHLIST_FILE
            try:
                watchlist_file = tmp / "watchlist.json"
                watchlist_file.write_text(json.dumps(["AAAUSDT", "BBBUSDT"]), encoding="utf-8")
                watchlist_top_gainer_model.top_gainer_critic.WATCHLIST_FILE = watchlist_file
                summary = watchlist_top_gainer_model.backfill_from_local_reports(
                    dataset_path=dataset,
                    report_dir=reports,
                    teacher_key="proxy_key",
                )
            finally:
                watchlist_top_gainer_model.top_gainer_critic.WATCHLIST_FILE = old_watchlist

            self.assertEqual(summary["reports_processed"], 1)
            self.assertEqual(summary["rows_annotated_total"], 2)

            written = [
                json.loads(line)
                for line in dataset.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            rows_by_sym = {row["sym"]: row for row in written}
            self.assertTrue(rows_by_sym["AAAUSDT"]["teacher"]["proxy_key"]["watchlist_top_gainer"])
            self.assertFalse(rows_by_sym["BBBUSDT"]["teacher"]["proxy_key"]["watchlist_top_gainer"])

    def test_train_and_evaluate_runs_on_synthetic_proxy_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset = tmp / "ml_dataset.jsonl"

            rows = []
            base_dt = datetime(2026, 4, 1, 8, 0, tzinfo=timezone.utc)
            teacher_key = "proxy_key"
            for day_offset in range(5):
                day_dt = base_dt + timedelta(days=day_offset)
                target_day = day_dt.date().isoformat()
                for sym, positive in (("AAAUSDT", True), ("BBBUSDT", False)):
                    for hour in (8, 18):
                        ts = day_dt.replace(hour=hour).strftime("%Y-%m-%dT%H:%M:%SZ")
                        row = _make_record(sym, ts, positive=positive)
                        row["teacher"] = {
                            teacher_key: {
                                "target_day_local": target_day,
                                "timezone": "Europe/Budapest",
                                "watchlist_top_gainer": positive,
                            }
                        }
                        rows.append(row)

            dataset.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                encoding="utf-8",
            )

            report = watchlist_top_gainer_model.train_and_evaluate(
                dataset,
                teacher_key=teacher_key,
                timezone_name="Europe/Budapest",
                max_local_hour=22,
                checkpoint_hours=(8, 18, 22),
                top_n=1,
                min_rows=8,
                min_days=3,
            )

            self.assertIn(report["chosen_model"], {"logistic", "mlp"})
            self.assertEqual(report["days_total"], 5)
            self.assertGreaterEqual(report["test_ranking"]["primary_score"], 0.0)
            checkpoint_hours = [item["hour_local"] for item in report["test_ranking"]["checkpoint_metrics"]]
            self.assertEqual(checkpoint_hours, [8, 18, 22])


if __name__ == "__main__":
    unittest.main()
