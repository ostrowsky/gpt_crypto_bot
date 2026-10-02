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
    settings.pop('LOCAL_LOGICAL_POLICY_ROLLOUT_ENABLED', None)
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
    local = getattr(config, 'LOCAL_LOGICAL_POLICY_ROLLOUT_ENABLED', False)
    if not local and not getattr(config, 'CERTIFIED_RULE_SCORE_POLICY_ENABLED', False):
        return 0.0
    try:
        root, key = release.ROOT, os.environ.get('RANKER_EVALUATOR_KEY', '').encode()
        if local:
            import json
            from logical_learning_authority import material
            base = Path(__file__).resolve().parents[1]/'.runtime/learning_roles_local'
            deployment = json.loads((base/'deployment.json').read_bytes())
            _, key = material(deployment)
            root = Path(deployment['release_root'])
        selected = release.select(root, champion_bytes(), kwargs['sym'], key, enabled=True)
        if selected is None:
            return 0.0
        model, ticket = selected
        if local and ticket.get('authority_mode') != 'logical_same_user':
            return 0.0
        from ml_candidate_ranker import build_runtime_candidate_record, predict_components_from_candidate_payload
        args = dict(kwargs)
        args['bar_ts'] = int(args['data']['t'][args['i']])
        args['btc_vs_ema50'] = float(getattr(config, '_btc_vs_ema50', 0))
        args.setdefault('near_miss', False)
        record = build_runtime_candidate_record(**args)
        components = predict_components_from_candidate_payload(model, record)
        value = probability_bonus(components['quality_proba'])
        if local:
            from forward_evidence_service import atomic
            # Actual score consumption, not a BUY/fill or proof of improvement.
            receipt = {'contract': 'bounded-score-consumption-v1', 'sym': args['sym'],
                       'tf': args['tf'], 'bar_ts': args['bar_ts'], 'bonus': value,
                       'candidate_sha256': ticket['candidate_sha256'],
                       'ticket_sha256': release.sha(release.canonical(ticket)),
                       'scope': 'score_only_not_buy_or_fill', 'closed_loop': False}
            atomic(root/'receipts'/(release.sha(release.canonical(receipt))+'.json'), receipt)
        return value
    except Exception:
        return 0.0
