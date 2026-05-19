from __future__ import annotations

import unittest

from audit_v2_v1_market_structure_features import _is_reasonable_ts_ms, _is_v1_structure_feature, _seq_feature_value


class V1MarketStructureFeatureAuditTest(unittest.TestCase):
    def test_identifies_v1_structure_names(self) -> None:
        self.assertTrue(_is_v1_structure_feature("btc_vs_ema50"))
        self.assertTrue(_is_v1_structure_feature("seq_trend_adx"))
        self.assertTrue(_is_v1_structure_feature("projected_leader_score_trend"))
        self.assertFalse(_is_v1_structure_feature("hour_sin"))

    def test_reasonable_ts_filter_rejects_malformed_values(self) -> None:
        self.assertTrue(_is_reasonable_ts_ms(1779105600000))
        self.assertFalse(_is_reasonable_ts_ms(999999999999999999))

    def test_sequence_feature_value_supports_model_schema(self) -> None:
        seq = [[1.0, 10.0], [2.0, 12.0], [4.0, 15.0]]
        names = ["close_norm", "adx"]
        self.assertEqual(_seq_feature_value("seq_last_adx", seq, names), 15.0)
        self.assertEqual(_seq_feature_value("seq_trend_adx", seq, names), 5.0)
        self.assertAlmostEqual(_seq_feature_value("seq_mean_close_norm", seq, names), 7.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
