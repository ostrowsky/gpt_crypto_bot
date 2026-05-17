from __future__ import annotations

import unittest


class TestV2TeacherConfidence(unittest.TestCase):
    def _label(self, mfe: float, state: str = "confirmed_trend"):
        from v2.lifecycle_labeling import LifecycleLabel
        from v2.state import SymbolState

        return LifecycleLabel(
            symbol="AAAUSDT",
            timeframe="15m",
            open_ts_ms=1,
            local_day="2026-05-17",
            state=SymbolState(state),
            label_version="hindsight_lifecycle_v1",
            day_open=100.0,
            day_mfe_pct=mfe,
            peak_index=5,
            confirmation_index=2,
        )

    def test_stronger_move_has_higher_confidence(self) -> None:
        from v2.teacher_confidence import score_label

        weak = score_label(self._label(3.1), bars_in_day=10).value
        strong = score_label(self._label(5.0), bars_in_day=10).value
        self.assertGreater(strong, weak)

    def test_noise_is_low_confidence(self) -> None:
        from v2.teacher_confidence import score_label

        noise = score_label(self._label(1.0, state="noise"), bars_in_day=10).value
        trend = score_label(self._label(5.0), bars_in_day=10).value
        self.assertLess(noise, trend)

    def test_emerging_move_is_less_confident_than_confirmed_trend(self) -> None:
        from v2.teacher_confidence import score_label

        emerging = score_label(self._label(5.0, state="emerging_move"), bars_in_day=10).value
        confirmed = score_label(self._label(5.0), bars_in_day=10).value
        self.assertLess(emerging, confirmed)

    def test_confidence_is_bounded(self) -> None:
        from v2.teacher_confidence import score_label

        value = score_label(self._label(99.0), bars_in_day=10).value
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)
