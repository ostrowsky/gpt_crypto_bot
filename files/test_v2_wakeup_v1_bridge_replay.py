from __future__ import annotations
import json, tempfile, unittest
from pathlib import Path

class TestV2WakeupV1BridgeReplay(unittest.TestCase):
    def test_bridge_selects_v1_structural_row_after_v2_wakeup(self):
        import replay_v2_wakeup_v1_bridge as mod
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); reports=root/'reports'; reports.mkdir(); events=root/'events.jsonl'; critic=root/'critic.jsonl'; watch=root/'watch.json'
            watch.write_text(json.dumps(['AAAUSDT','BBBUSDT']),encoding='utf-8')
            (reports/'top_gainer_critic_2026-05-01_final.json').write_text(json.dumps({'target_day_local':'2026-05-01','summary':{'watchlist_top_denominator':'exchange_top_filtered_to_watchlist'},'watchlist_top_gainers':[{'symbol':'AAAUSDT','status':'bought','latest_exit_pnl_pct':1.0}]}),encoding='utf-8')
            events.write_text('\n'.join(json.dumps(x) for x in [
                {'ts':'2026-05-01T01:00:00Z','sym':'AAAUSDT','state':'emerging_move','bootstrap':False},
                {'ts':'2026-05-01T01:00:00Z','sym':'BBBUSDT','state':'emerging_move','bootstrap':False},
            ]),encoding='utf-8')
            def row(sym, ret):
                return {'sym':sym,'ts_signal':'2026-05-01T02:00:00Z','signal_type':'trend','decision':{'signal_flags':{'entry_ok':True}},'f':{'close_vs_ema20':1,'macd_hist_norm':0.1,'rsi':60,'daily_range':5,'slope':0.5,'vol_x':1.5},'labels':{'ret_5':ret}}
            critic.write_text('\n'.join(json.dumps(x) for x in [row('AAAUSDT',1.2),row('BBBUSDT',-0.5)]),encoding='utf-8')
            payload=mod.run_replay(reports_dir=reports,events_file=events,critic_dataset=critic,watchlist_file=watch,save=False)
        m=payload['policies']['v2_wakeup_v1_structural']
        self.assertEqual(m['n'],2)
        self.assertEqual(m['top_count'],1)
        self.assertEqual(m['top_precision_pct'],50.0)
        self.assertEqual(m['ret5_precision_pct'],50.0)

if __name__=='__main__': unittest.main()
