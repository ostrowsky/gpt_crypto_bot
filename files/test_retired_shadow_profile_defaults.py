from __future__ import annotations

import unittest

import config


class RetiredShadowProfileDefaultsTests(unittest.TestCase):
    def test_forward_failed_profiles_are_disabled_by_default(self) -> None:
        self.assertFalse(config.V2_BTC_TREND_WATCH_TELEGRAM_ENABLED)
        self.assertFalse(config.OBSERVABLE_TAIL_SHADOW_ENABLED)


if __name__ == "__main__":
    unittest.main()
