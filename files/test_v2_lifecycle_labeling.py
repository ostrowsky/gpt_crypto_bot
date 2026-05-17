from __future__ import annotations

import unittest


def _bars(closes, highs=None):
    from v2.history import CanonicalBar

    highs = highs or closes
    out = []
    for i, (close, high) in enumerate(zip(closes, highs)):
        out.append(CanonicalBar("AAAUSDT", "15m", 1_779_000_000_000 + i * 900_000, closes[0], high, close, close, 1.0))
    return out


class TestV2LifecycleLabeling(unittest.TestCase):
    def test_labels_full_trend_lifecycle(self) -> None:
        from v2.lifecycle_labeling import label_bars
        from v2.state import SymbolState

        bars = _bars(
            [100, 101, 102, 103, 104, 104.5, 104.2, 103.0, 101.0, 99.5],
            [100, 101.5, 102.2, 103.2, 104.2, 105.0, 104.6, 103.4, 101.3, 100.0],
        )
        states = [label.state for label in label_bars(bars)]
        self.assertEqual(states[0], SymbolState.EMERGING_MOVE)
        self.assertEqual(states[2], SymbolState.CONFIRMED_TREND)
        self.assertIn(SymbolState.MATURE_TREND, states)
        self.assertIn(SymbolState.EXHAUSTION, states)
        self.assertEqual(states[-1], SymbolState.REVERSAL)

    def test_nonqualifying_day_is_noise(self) -> None:
        from v2.lifecycle_labeling import label_bars
        from v2.state import SymbolState

        states = [label.state for label in label_bars(_bars([100, 100.5, 101.0, 100.7]))]
        self.assertEqual(set(states), {SymbolState.NOISE})

    def test_summary_reports_no_invalid_transitions(self) -> None:
        from v2.lifecycle_labeling import label_bars, summarize_labels

        labels = label_bars(
            _bars(
                [100, 101, 102, 103, 104, 104.5, 104.2, 103.0, 101.0, 99.5],
                [100, 101.5, 102.2, 103.2, 104.2, 105.0, 104.6, 103.4, 101.3, 100.0],
            )
        )
        self.assertEqual(summarize_labels(labels)["invalid_transition_counts"], {})
