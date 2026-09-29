import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np

from research_rocket_capture import pair_events, capture_day, bounds, policy, blocks, DAY, complete_series, post_entry_retention, strict_cached_series, audit_actual


def event(kind, hour, **kw):
    return dict(event=kind,ts=f"2026-09-24T{hour}:00Z",sym="QNTUSDT",tf="15m",source_file="bot",**kw)


class RocketTests(unittest.TestCase):
    def test_pairs_repeated_trades_not_first_last(self):
        events=[event("entry","10:00",price=100),event("exit","11:00",entry_price=100,exit_price=110),
                event("entry","12:00",price=120),event("exit","13:00",entry_price=120,exit_price=115)]
        pairs, problems=pair_events(events)
        r=capture_day(dict(day="2026-09-24",symbol="QNTUSDT",day_open=100,day_close=125,day_change_pct=25),pairs)
        self.assertFalse(problems)
        self.assertEqual(r["capture_pct"],20)
        self.assertEqual(r["price_gain_sum"],5)

    def test_ambiguous_and_wrong_price_not_matched(self):
        events=[event("entry","10:00",price=100),event("entry","10:15",price=101),
                event("exit","11:00",entry_price=101,exit_price=110)]
        pairs,problems=pair_events(events)
        self.assertEqual(pairs,[])
        self.assertEqual(len(problems),2)
        pairs,_=pair_events([event("entry","10:00",price=100),event("exit","11:00",entry_price=90,exit_price=110)])
        self.assertEqual(pairs,[])

    def test_duplicate_event_and_source_isolation(self):
        e=event("entry","10:00",price=100)
        x=event("exit","11:00",entry_price=100,exit_price=110)
        self.assertEqual(len(pair_events([e,e,x])[0]),1)
        x["source_file"]="agent"
        self.assertEqual(pair_events([e,x])[0],[])

    def test_missing_and_crossday_unknown(self):
        rocket=dict(day="2026-09-24",symbol="QNTUSDT",day_open=100,day_close=125,day_change_pct=25)
        self.assertIsNone(capture_day(rocket,[])["capture_pct"])
        start,end=bounds(rocket["day"])
        p=dict(symbol="QNTUSDT",entry_ts=start-1,exit_ts=end-1,entry_price=100,exit_price=110)
        self.assertEqual(capture_day(rocket,[p])["status"],"cross_day_position")

    def test_gap_rejected(self):
        d=np.array([(0,),(100,),(300,)],dtype=[("t","i8")])
        self.assertFalse(complete_series(d,0,400,100))

    def test_peak_before_entry_excluded(self):
        data=np.array([(0,1000.),(100,120.),(200,115.)],dtype=[("t","i8"),("h","f8")])
        trade=dict(entry_ts=100,entry_price=100,exit_price=110)
        self.assertEqual(post_entry_retention(trade,data,300,100)["retention_pct"],50)
        self.assertIsNone(post_entry_retention(trade,data,400,100)["retention_pct"])

    def test_conflicting_caches_are_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            a,b=Path(directory)/"a.json",Path(directory)/"b.json"
            a.write_text(json.dumps([dict(t=0,o=100,h=110,l=90,c=105,v=10)]))
            b.write_text(json.dumps([dict(t=0,o=100,h=110,l=90,c=106,v=10)]))
            data,q=strict_cached_series({("X","15m"):[(0,900000,a),(0,900000,b)]},"X","15m",0,900000)
            self.assertIsNone(data)
            self.assertEqual(q["conflicting_timestamps"],1)

    def test_short_read_does_not_spin(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            for name in ("bot_events.jsonl","agent_events.jsonl"):
                (Path(directory)/name).write_text("")
            with patch.object(Path,"stat",return_value=SimpleNamespace(st_size=100)):
                result=audit_actual([],Path(directory))
            self.assertTrue(all(s["coverage"]=="truncated_during_read" for s in result["sources"]))

    def test_split_chronological(self):
        schedule=blocks(0,180*DAY)
        self.assertEqual([r[2] for r in schedule],["discovery"]*3+["validation"]+["holdout"]*2)

    def test_policy_restores_config_and_isolates_rules(self):
        import replay_backtest as rb
        import config
        orig=rb._chase_guard_reason_for_replay_variant
        with policy("cluster_off"):
            self.assertIs(rb._chase_guard_reason_for_replay_variant,orig)
            self.assertEqual(rb._signal_cluster_cap_replay("15m_impulse"),10)
        self.assertIs(rb._chase_guard_reason_for_replay_variant,orig)
        threshold=config.RSI_OVERBOUGHT
        with self.assertRaises(RuntimeError):
            with policy("chase_off"):
                self.assertIsNone(rb._chase_guard_reason_for_replay_variant())
                raise RuntimeError()
        self.assertIs(rb._chase_guard_reason_for_replay_variant,orig)
        self.assertEqual(config.RSI_OVERBOUGHT,threshold)

    def test_rsi_disabled_exposes_later_hard_exit(self):
        import replay_backtest as rb
        feat={"ema_fast":np.array([90.,90.]),"rsi":np.array([90.,90.]),
              "adx":np.array([30.,30.]),"slope":np.array([-1.,-1.])}
        with policy("rsi_exit_off"):
            result=rb.check_exit_conditions(feat,1,np.array([100.,100.]))
        self.assertIn("slope",result)


if __name__=="__main__":
    unittest.main()
