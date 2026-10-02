"""Bounded certified BUY score policy, independent of legacy ranker enablement."""
import os
from pathlib import Path

import config
import policy_provenance
import validated_ranker_rollout as release

CONTRACT = 'certified-rule-score-policy-v1'
SOURCES = ('monitor.py', 'replay_backtest.py', 'strategy.py', 'indicators.py',
           'config.py', 'ml_candidate_ranker.py', 'policy_provenance.py',
           'certified_rule_score_policy.py')


def champion_bytes():
    root = Path(__file__).resolve().parent
    settings = policy_provenance._safe_config_snapshot()
    settings.pop('CERTIFIED_RULE_SCORE_POLICY_ENABLED', None)
    return release.canonical({'contract': CONTRACT,
        'sources': {name: release.sha((root/name).read_bytes()) for name in SOURCES},
        'models': {name: release.sha((root/name).read_bytes())
                   for name in ('ml_signal_model.json', 'ml_candidate_ranker.json')},
        'config': settings})


def probability_bonus(value):
    import math
    p = float(value)
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError('invalid candidate probability')
    return release.bounded_bonus(2*(p-0.5))


def bonus(**kwargs):
    if not getattr(config, 'CERTIFIED_RULE_SCORE_POLICY_ENABLED', False):
        return 0.0
    try:
        selected = release.select(release.ROOT, champion_bytes(), kwargs['sym'],
            os.environ.get('RANKER_EVALUATOR_KEY', '').encode(), enabled=True)
        if selected is None:
            return 0.0
        model, _ = selected
        from ml_candidate_ranker import build_runtime_candidate_record, predict_components_from_candidate_payload
        args = dict(kwargs)
        args['bar_ts'] = int(args['data']['t'][args['i']])
        args['btc_vs_ema50'] = float(getattr(config, '_btc_vs_ema50', 0))
        args.setdefault('near_miss', False)
        record = build_runtime_candidate_record(**args)
        components = predict_components_from_candidate_payload(model, record)
        return probability_bonus(components['quality_proba'])
    except Exception:
        return 0.0
