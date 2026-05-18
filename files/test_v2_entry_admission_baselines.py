from __future__ import annotations

import unittest

from audit_v2_entry_admission_baselines import _admit


class EntryAdmissionBaselinesTest(unittest.TestCase):
    def test_combined_rule_requires_all_enabled_conditions(self) -> None:
        row = {
            "belief": {"emerging_move": 0.45, "confirmed_trend": 0.20},
            "v1_projected_structural": {"projected_leader_score_trend": 6.0},
            "v1_temporal": {"prior_structural_scout": True},
        }
        self.assertTrue(_admit(row, 0.60, 5.0, True))
        self.assertFalse(_admit(row, 0.70, 5.0, True))
        self.assertFalse(_admit(row, 0.60, 8.0, True))
        row["v1_temporal"]["prior_structural_scout"] = False
        self.assertFalse(_admit(row, 0.60, 5.0, True))


if __name__ == "__main__":
    unittest.main()
