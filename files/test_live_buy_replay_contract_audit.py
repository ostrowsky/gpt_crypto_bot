import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import audit_live_buy_replay_contract as audit


class AuditTests(unittest.TestCase):
    def test_real_conditional_contract_probes_fail_not_full_live_pass(self):
        rows=asyncio.run(audit.probes())
        self.assertEqual(rows[0]['live'],16)
        self.assertEqual(rows[0]['replay'],48)
        self.assertEqual([r['state'] for r in rows],['FAIL','FAIL'])
        self.assertFalse(rows[1]['live_allowed'])
        self.assertTrue(rows[1]['replay_allowed'])

    def test_costs_turn_small_gross_winner_into_net_loss(self):
        row={'entry_price':100.,'exit_price':100.1,'bars_held':1}
        result=audit.summarize([row],7.5,5.)
        self.assertEqual(result['n'],1)
        self.assertEqual(result['gross_positive'],1)
        self.assertEqual(result['net_positive'],0)
        self.assertEqual(result['gross_winners_lost_to_costs'],1)
        self.assertLess(result['net_trade_mean_pct'],0)
        self.assertAlmostEqual(audit.summarize([row],0,0)['net_trade_mean_pct'],.1)

    def test_empty_is_unknown_and_partial_is_not_simplified(self):
        self.assertIsNone(audit.summarize([],7.5,5)['gross_mean_pct'])
        with self.assertRaises(ValueError):
            audit.summarize([{'partial_exit_taken':True}],7.5,5)
        for entry, exit in ((0,100),(100,float('nan'))):
            with self.assertRaisesRegex(ValueError,'execution price'):
                audit.summarize([{'entry_price':entry,'exit_price':exit}],7.5,5)
        with self.assertRaisesRegex(ValueError,'execution costs'):
            audit.summarize([],float('nan'),5)

    def test_reason_and_output_preservation(self):
        self.assertEqual(audit.exit_class('portfolio replacement'), 'replacement')
        self.assertEqual(audit.exit_class('time (48 bars)'), 'time')
        self.assertEqual(audit.exit_class('open_at_end'), 'boundary')
        self.assertEqual(audit.exit_class('price below EMA20 (123.45)'), 'ema20')
        self.assertEqual(audit.exit_class('2 closes below EMA20 (1.23)'), 'two_closes_below_ema20')
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'result.json'
            p.touch()
            with patch.object(audit,'verify') as verify:
                with self.assertRaisesRegex(ValueError,'overwrite'):
                    asyncio.run(audit.audit(Path(tmp),p))
                verify.assert_not_called()

    def test_invalid_receipt_never_reaches_numeric_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch.object(Path,'read_bytes',return_value=b'{"data.json":"bad"}'):
                with self.assertRaisesRegex(ValueError,'receipt'):
                    audit.verify(root)

    def test_policy_source_and_maximum_bounds_drift_fail_closed(self):
        root=Path('audit_fixture')
        def read_bytes(path):
            if path.name == 'receipt.json': return b'{}'
            if path.name == 'registration.json':
                return json.dumps({'sources':{'monitor.py':'bad'},
                                   'maximum_available_bounds':[1,2]}).encode()
            return b'policy'
        with patch.object(Path,'read_bytes',read_bytes):
            with self.assertRaisesRegex(ValueError,'policy source drift'):
                audit.verify(root)
        def bounds_bytes(path):
            if path.name == 'receipt.json': return b'{}'
            if path.name == 'registration.json':
                return b'{"sources":{},"maximum_available_bounds":[1,2]}'
            return b'{"start_ms":1,"end_ms":3}'
        with patch.object(Path,'read_bytes',bounds_bytes):
            with self.assertRaisesRegex(ValueError,'maximum period mismatch'):
                audit.verify(root)


if __name__ == '__main__': unittest.main()
