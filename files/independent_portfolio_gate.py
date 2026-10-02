"""Read-only independent account recomputation; unsigned trainer summaries never approve."""
from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys
import time

from portfolio_alpha import evaluate_portfolio_alpha, _simulate_account
from validated_ranker_rollout import CONTRACT, canonical, seal, sha, unseal

DAY = 86400000
STEP = 900000


def source_hash():
    # Shared account implementation is part of the independent numeric contract.
    names = ('independent_portfolio_gate.py', 'independent_portfolio_confirmation.py',
             'portfolio_alpha.py', 'validated_ranker_rollout.py',
             'monitor.py', 'ml_candidate_ranker.py', 'config.py', 'strategy.py',
             'replay_backtest.py', 'indicators.py', 'policy_provenance.py', 'process_lock.py')
    return sha(canonical({name: sha(Path(__file__).with_name(name).read_bytes()) for name in names}))


def harness_passes(project_root=None):
    harness = (Path(project_root)/'files'/'truth_harness.py' if project_root is not None
               else Path(__file__).with_name('truth_harness.py'))
    result = subprocess.run([sys.executable, str(harness), 'full'],
                            capture_output=True, timeout=120)
    return result.returncode == 0


def evaluate_bundle(raw, attestation, authority_key, candidate_sha, champion_sha, now):
    cert = unseal(attestation, authority_key)
    if (cert.get('bundle_sha256') != sha(raw)
            or cert.get('candidate_sha256') != candidate_sha
            or cert.get('champion_sha256') != champion_sha
            or cert.get('evaluator_sha256') != source_hash()
            or not cert.get('issued_at', math.inf) <= now < cert.get('expires_at', 0)
            or cert['expires_at']-cert['issued_at'] > 86400
            or cert.get('contract') != 'independent-market-policy-certification-v1'
            or any(cert.get(k) is not True for k in
                   ('point_in_time_universe', 'raw_closed_provenance', 'live_policy_parity',
                    'no_trainer_holdout_access', 'operator_experiment_approved'))):
        raise ValueError('missing independent population/policy certification')
    bundle = json.loads(raw)
    if bundle.get('phase') == 'canary' and cert.get('actual_assignment_verified') is not True:
        raise ValueError('canary assignment not certified')
    result = compare_accounts(bundle, now)
    return dict(result, bundle_sha256=sha(raw), certification_sha256=sha(canonical(attestation)))


def compare_accounts(bundle, now):
    """Numerical comparison only; this function never certifies provenance or release."""
    phase = bundle['phase']
    if phase not in ('historical', 'sealed', 'shadow', 'canary'):
        raise ValueError('unknown evidence phase')
    start, end = bundle['start_ms'], bundle['end_ms']
    if (not isinstance(start, int) or not isinstance(end, int) or start % STEP or end % STEP
            or end <= start or end/1000 > now):
        raise ValueError('invalid evaluation window')
    if phase == 'historical' and [start, end] != bundle['maximum_available_bounds']:
        raise ValueError('historical period shortened')
    exposure = bundle['last_model_exposure_ms']
    if not isinstance(exposure, int) or (phase != 'historical' and start < exposure+2*DAY):
        raise ValueError('unsealed model exposure')
    if end-start < 30*DAY:
        raise ValueError('insufficient observation period')
    prices = bundle['prices']
    universe = bundle['universe']
    if (not universe or len(universe) != len(set(universe)) or set(prices) != set(universe)
            or 'BTCUSDT' not in prices):
        raise ValueError('partial population')
    expected = list(range(start, end+1, STEP))
    for rows in prices.values():
        if [r[0] for r in rows] != expected or any(
                not math.isfinite(float(r[1])) or float(r[1]) <= 0 for r in rows):
            raise ValueError('invalid closed valuation grid')
    fee, slip = float(bundle['fee_bps']), float(bundle['slippage_bps'])
    if not math.isfinite(fee) or not math.isfinite(slip) or fee < 7.5 or slip < 5:
        raise ValueError('insufficient costs')
    reports, curves = {}, {}
    for arm in ('champion', 'candidate'):
        trades = bundle[arm+'_trades']
        if len(trades) < 100:
            raise ValueError('insufficient trades')
        for trade in trades:
            if (trade['sym'] not in universe or not start <= trade['entry_ts'] <= trade['exit_ts'] <= end
                    or trade['entry_ts'] % STEP or trade['exit_ts'] % STEP
                    or any(not math.isfinite(float(trade[k])) or float(trade[k]) <= 0
                           for k in ('entry_price', 'exit_price'))
                    or trade.get('partial_exit_taken', False)):
                raise ValueError('unsupported or noncausal trade')
        report = evaluate_portfolio_alpha(trades, price_series_by_symbol=prices,
            benchmark_series=prices['BTCUSDT'], window_start_ms=start, window_end_ms=end,
            requested_days=(end-start)//DAY, universe=universe, variant=arm,
            capacity=10, fee_bps=fee, slippage_bps=slip)
        if (report['status'] != 'complete' or report['coverage']['valuation_coverage'] != 1
                or report['coverage']['benchmark_grid']['missing']
                or report['coverage']['contract_violations']):
            raise ValueError('incomplete account evaluation')
        reports[arm] = report
        account = _simulate_account(trades, price_series_by_symbol=prices,
            valuation_timestamps=expected, capacity=10, initial_capital=10000,
            fee_bps=fee, slippage_bps=slip)
        curves[arm] = dict(account.equity_curve)
    left, right = [reports[k]['portfolio'] for k in ('champion', 'candidate')]
    delta = right['net_return_after_costs_pct']-left['net_return_after_costs_pct']
    dd = right['max_drawdown_after_costs_pct']
    # Paired daily portfolio returns, not per-trade PnL. Seven-day moving blocks
    # preserve short-range dependence; deterministic seed prevents retry lottery.
    import numpy as np
    daily = []
    for t in range(start+DAY, end+1, DAY):
        changes = [100*(curves[arm][t]/curves[arm][t-DAY]-1) for arm in ('champion', 'candidate')]
        daily.append(changes[1]-changes[0])
    ci = None
    if len(daily) >= 30:
        rng = np.random.default_rng(20261001)
        blocks = [daily[i:i+7] for i in range(len(daily)-6)]
        draws = []
        for _ in range(2000):
            sample = [v for i in rng.integers(0, len(blocks), size=(len(daily)+6)//7) for v in blocks[i]]
            draws.append(float(np.mean(sample[:len(daily)])))
        ci = [float(v) for v in np.quantile(draws, [.025, .975])]
    passed = (ci is not None and ci[0] > 0 and delta > 0 and dd <= 15 and dd <= left['max_drawdown_after_costs_pct']+1
              and reports['candidate']['net_alpha_after_costs'] >= 0)
    return {'phase': phase, 'start_ms': start, 'end_ms': end,
            'portfolio_delta_pp': delta, 'reports': reports, 'passed': passed,
            'paired_daily_95ci_pp': ci, 'paired_days': len(daily),
            'uncertainty': 'seven-day moving-block bootstrap; not a guarantee of future return'}


def authorize(evidence, authority_key, evaluator_key, candidate_raw, champion_raw,
              stage='CANARY', now=None, harness_root=None):
    now = time.time() if now is None else now
    if stage not in ('CANARY', 'PROMOTED'):
        raise ValueError('unsupported stage')
    required = ['historical', 'sealed', 'shadow']+(['canary'] if stage == 'PROMOTED' else [])
    if set(evidence) != set(required):
        raise ValueError('missing separate historical/sealed/forward evidence')
    reports = []
    for phase in required:
        raw, cert = evidence[phase]
        report = evaluate_bundle(raw, cert, authority_key, sha(candidate_raw), sha(champion_raw), now)
        if report['phase'] != phase or not report['passed']:
            raise ValueError('independent portfolio gate rejected: '+phase)
        reports.append(report)
    # Forward samples cannot be re-used for two gate stages.
    chronological = [r for r in reports if r['phase'] != 'historical']
    for left, right in zip(chronological, chronological[1:]):
        if right['start_ms'] < left['end_ms']:
            raise ValueError('overlapping sealed/shadow/canary cohorts')
    if now-chronological[-1]['end_ms']/1000 > 86400:
        raise ValueError('latest forward evidence is stale')
    if not (harness_passes() if harness_root is None else harness_passes(harness_root)):
        raise ValueError('full Truth Harness FAIL/UNKNOWN blocks activation')
    body = {'contract': CONTRACT, 'stage': stage, 'issued_at': now, 'expires_at': now+86400,
            'candidate_sha256': sha(candidate_raw), 'champion_sha256': sha(champion_raw),
            'evaluator_sha256': source_hash(), 'max_bonus': 1.0,
            'fraction': 0.05 if stage == 'CANARY' else 1.0,
            'evidence': [{k:r[k] for k in ('phase', 'bundle_sha256', 'certification_sha256',
                                          'portfolio_delta_pp', 'paired_daily_95ci_pp',
                                          'paired_days', 'reports', 'uncertainty')} for r in reports]}
    return seal(body, evaluator_key)


if __name__ == '__main__':
    import argparse
    import os
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    request = json.loads(args.request.read_bytes())
    evidence = {phase: (Path(item['bundle']).read_bytes(),
                         json.loads(Path(item['certification']).read_bytes()))
                for phase, item in request['evidence'].items()}
    ticket = authorize(evidence, os.environ.get('RANKER_COVERAGE_AUTHORITY_KEY', '').encode(),
                       os.environ.get('RANKER_EVALUATOR_KEY', '').encode(),
                       Path(request['candidate']).read_bytes(), Path(request['champion']).read_bytes(),
                       request.get('stage', 'CANARY'))
    with args.output.open('xb') as handle:
        handle.write(canonical(ticket))
