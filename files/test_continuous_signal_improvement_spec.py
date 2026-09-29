"""Documentation contract checks; these do not prove runtime implementation."""
from pathlib import Path
import unittest


class ContinuousSignalImprovementSpecTests(unittest.TestCase):
    def test_required_contracts_are_explicit(self):
        root = Path(__file__).resolve().parents[1]
        spec = (root / 'docs/specs/continuous-signal-improvement.md').read_text(encoding='utf-8')
        for requirement in (
            'implementation pending', 'unacceptable incomplete learning loop',
            'independent evaluation', 'sealed holdout', 'maximum available',
            'after fees/slippage', 'last-known-good', 'rollback',
            'morning report MUST', 'aged_label_coverage', 'TH-01 through TH-12',
            'does not enable BUY/SELL changes',
        ):
            with self.subTest(requirement=requirement):
                self.assertIn(requirement, spec)

    def test_spec_is_registered(self):
        root = Path(__file__).resolve().parents[1]
        index = (root / 'docs/FEATURE_SPEC_INDEX.md').read_text(encoding='utf-8')
        self.assertIn('docs/specs/continuous-signal-improvement.md', index)


if __name__ == '__main__':
    unittest.main()
