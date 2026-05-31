# Learning Loop Architecture Roadmap

Date: 2026-05-31
Status: architecture proposal

## Principle

Every major target metric should have three layers:

1. measurement: daily labels and attribution;
2. shadow learner: model/policy recommendation without trading side effects;
3. replay/promotion gate: production adoption only after robust evidence.

## Current Gaps

### Entry / early capture

Current: V1 rules, ML ranker, V2 shadow radar.

Learning loop exists but is incomplete: V2 has high recall and low precision. Next learning loop should focus on calibrated admission precision, not more raw alerts.

### Selection / portfolio cap

Current: score gates and ranker training.

Learning loop exists partially through ML candidate ranker. Missing: explicit EV/risk calibration under 10-slot portfolio cap and replacement uplift attribution.

### Blocked winners

Current: structured blocked logging and critic.

Learning loop is measurement-heavy, not adaptive. Missing: per-blocker harm model that proposes candidate relaxations in shadow/replay only.

### Exit monetization

Current: V1 rule-based SELL, exit auditor, exit discriminator, suspicious re-entry replay.

Learning loop did not affect live behavior until shadow re-entry alerts. Next: daily scoring of shadow re-entry precision and forward returns.

### Risk / exposure

Current: portfolio cap and group cap.

Missing: correlation/exposure learner in shadow mode, then replay-gated cap adjustments.

## Recommended Order

1. Shadow suspicious re-entry alerts.
2. Daily scorecard for shadow re-entry outcomes.
3. Calibrated V2 admission precision learner.
4. Portfolio EV/risk ranker shadow report.
5. Blocker harm model.
6. Correlation exposure shadow learner.
