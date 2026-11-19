"""RFP09 decision denominators are not fill counts or achievement claims."""
import unittest
from binance_demo_bot import receipt_flow_portfolio_report as report


class DecisionReportContracts(unittest.TestCase):
    def test_RFP09_selected_decisions_are_separate_from_fills(self):
        def e(kind,**kw):return dict(kind=kind,evidence_kind='SIMULATION',**kw)
        result=report.journal_reasons(iter([
            e('DECISION',selected={'symbol':'BTCUSDT'},reason=None),
            e('BUY_EXPIRED',reason='TTL'),
            e('DECISION',selected=None,reason='NO_ADMISSIBLE_EDGE'),
            e('DECISION',selected=None,reason='NO_SIGNAL'),e('BUY_FILL')]))
        self.assertEqual(result['decision_N'],3)
        self.assertEqual(result['selected_decisions_n'],1)
        self.assertEqual(result['buy_fills_n'],1)
        self.assertEqual(result['decision_reason_counts'],{'SELECTED':1,'NO_ADMISSIBLE_EDGE':1,'NO_SIGNAL':1})
        self.assertAlmostEqual(result['selection_fraction'],1/3)

    def test_RFP09_empty_denominator_is_unknown(self):
        r=report.journal_reasons(iter([]))
        self.assertEqual(r['decision_N'],0);self.assertIsNone(r['selection_fraction'])

    def test_RFP09_mixed_actual_or_ambiguous_decision_is_rejected(self):
        with self.assertRaises(ValueError):report.journal_reasons(iter([
            dict(kind='DECISION',evidence_kind='ACTUAL_DEMO',selected=None,reason='NO_SIGNAL')]))
        with self.assertRaises(ValueError):report.journal_reasons(iter([
            dict(kind='DECISION',evidence_kind='SIMULATION',selected=None,reason=None)]))


if __name__=='__main__':unittest.main()
