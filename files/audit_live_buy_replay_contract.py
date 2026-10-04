"""Read-only, maximum-bundle contract probes and trade-loss diagnostics."""
import argparse
import ast
import asyncio
from collections import defaultdict
import ctypes
import json
import math
import re
from pathlib import Path
from statistics import mean, median
import time
from unittest.mock import patch

import numpy as np
import config
import monitor
import replay_backtest as rb
from validated_ranker_rollout import canonical, sha


def summarize(trades, fee_bps, slip_bps):
    fee, slip = fee_bps/10000, slip_bps/10000
    if not all(math.isfinite(v) and 0 <= v < 1 for v in (fee, slip)):
        raise ValueError('invalid execution costs')
    gross, net, bars = [], [], []
    for row in trades:
        if row.get('partial_exit_taken'):
            raise ValueError('partial trade needs its full cashflow contract')
        if not all(math.isfinite(row[k]) and row[k] > 0 for k in ('entry_price','exit_price')):
            raise ValueError('invalid execution price')
        ratio = row['exit_price']/row['entry_price']
        gross.append(100*(ratio-1))
        net.append(100*(ratio*(1-slip)/(1+slip)*(1-fee)/(1+fee)-1))
        bars.append(row.get('bars_held', 0))
    n = len(trades)
    return {'n': n, 'gross_mean_pct': mean(gross) if n else None,
            'gross_median_pct': median(gross) if n else None,
            'net_trade_mean_pct': mean(net) if n else None,
            'net_trade_median_pct': median(net) if n else None,
            'gross_positive': sum(v > 0 for v in gross),
            'net_positive': sum(v > 0 for v in net),
            'gross_winners_lost_to_costs': sum(g > 0 and z <= 0 for g,z in zip(gross,net)),
            'held_at_most_one_bar': sum(v <= 1 for v in bars),
            'median_bars_held': median(bars) if n else None,
            'scope': 'unweighted_trade_diagnostics_not_portfolio_alpha'}


def exit_class(reason):
    reason = str(reason).lower()
    if 'boundary' in reason or reason == 'open_at_end': return 'boundary'
    if 'replac' in reason: return 'replacement'
    if 'atr' in reason or 'trail' in reason: return 'trailing'
    if 'weak' in reason: return 'weak'
    if reason.startswith('time'): return 'time'
    if 'rsi' in reason: return 'rsi'
    if 'ema20' in reason: return 'two_closes_below_ema20' if reason.startswith('2 ') else 'ema20'
    # Prices/thresholds are evidence fields, not separate strategy classes.
    return re.sub(r'\d+(?:\.\d+)?', '#', reason)


def live_hourly_hold():
    """Evaluate the actual live base assignment, not a copied policy formula."""
    tree = ast.parse(Path(monitor.__file__).read_text(encoding='utf-8'))
    poll = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == '_poll_coin')
    entry = next(n for n in ast.walk(poll) if isinstance(n, ast.If)
                 and isinstance(n.test, ast.Name) and n.test.id == 'entry_ok'
                 and any(isinstance(x, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'max_hold'
                         for t in x.targets) for x in n.body))
    assignment = next(n for n in entry.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'max_hold' for t in n.targets))
    return eval(compile(ast.Expression(assignment.value), '<live max_hold>', 'eval'),
                {'config':config, 'getattr':getattr, 'tf':'1h'}), assignment.lineno


async def probes():
    hold, line = live_hourly_hold()
    with patch.object(rb, 'check_entry_conditions', return_value=(True,'')), \
         patch.object(rb, 'check_breakout_conditions', return_value=(False,'')), \
         patch.object(rb, 'check_retest_conditions', return_value=(False,'')), \
         patch.object(rb, 'check_trend_surge_conditions', return_value=(False,'')), \
         patch.object(rb, 'check_impulse_conditions', return_value=(False,'')), \
         patch.object(rb, 'check_alignment_conditions', return_value=(False,'')), \
         patch.object(rb, 'get_effective_entry_mode', return_value=('trend',False)):
        replay_hold = rb._entry_candidate({}, 30, np.ones(64), '1h')[2]
    state = monitor.MonitorState()
    with patch.object(monitor, 'external_agent_symbol_count', return_value=10):
        live_allowed, reason = monitor._check_portfolio_limits('BTCUSDT', state)
    data = np.zeros(70, dtype=rb._KLINE_DTYPE)
    step = rb.BAR_MS['15m']
    data['t'] = 1774994400000 + np.arange(70)*step
    for key in ('o','h','l','c'): data[key] = 100.
    data['v'] = 10.
    feat = {'atr':np.ones(70)}
    frame = int(data['t'][60])+step
    candidate = rb.ReplayCandidate(sym='BTCUSDT',tf='15m',mode='breakout',ts_ms=frame,
        i=60,price=100.,trail_k=2.,max_hold_bars=6,score=100.,top_gainer_score=100.)
    replay_state = {}
    with patch.object(rb, '_load_temporal_scout_events', return_value=({},{})):
        await rb.simulate_portfolio(['BTCUSDT'],['15m'], {('BTCUSDT','15m'):(data,feat)},
            {'BTCUSDT':(data,feat)}, {}, None, max_open_positions=10,
            enable_replacement=False, replace_min_delta=8, variant='score_replace_cluster',
            top_gainer_score_min=34., candidate_snapshot=({frame:[candidate]},{frame},1),
            stream_state=replay_state)
    replay_allowed = bool(replay_state['open_positions'])
    return [{'name':'hourly_trend_base_hold', 'live':hold, 'replay':replay_hold,
             'state':'PASS' if hold == replay_hold else 'FAIL', 'live_line':line,
             'scope':'real_replay_picker_vs_live_source_assignment_not_full_poll'},
            {'name':'external_agent_fills_all_ten_slots', 'live_allowed':live_allowed,
             'replay_allowed':replay_allowed, 'live_reason':reason,
             'state':'PASS' if live_allowed == replay_allowed else 'FAIL',
             'scope':'conditional_capacity_gate_vs_real_simulator_not_observed_history'}]


def verify(root):
    receipt = json.loads((root/'receipt.json').read_bytes())
    for name,digest in receipt.items():
        path = (root/name).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path.read_bytes()) != digest:
            raise ValueError('receipt mismatch: '+name)
    reg = json.loads((root/'registration.json').read_bytes())
    for name,digest in reg['sources'].items():
        path = Path(__file__).with_name(name)
        if sha(path.read_bytes()) != digest: raise ValueError('policy source drift: '+name)
    bundle = json.loads((root/'historical_unsigned.json').read_bytes())
    if [bundle['start_ms'],bundle['end_ms']] != reg['maximum_available_bounds']:
        raise ValueError('maximum period mismatch')
    comparison = json.loads((root/'result.json').read_bytes())['comparison']
    for report in comparison['reports'].values():
        for name,digest in report['provenance']['source_hashes'].items():
            if sha(Path(__file__).with_name(name).read_bytes()) != digest:
                raise ValueError('account source drift: '+name)
    return bundle, comparison, sha(canonical(receipt))


async def audit(root, output):
    if output.exists(): raise ValueError('refusing to overwrite audit evidence')
    bundle, comparison, receipt_hash = verify(root)
    tests = await probes()
    trades = bundle['champion_trades']
    groups = defaultdict(list)
    for row in trades:
        groups['exit/'+exit_class(row['exit_reason'])].append(row)
        groups['mode/'+row['tf']+'/'+row['mode']].append(row)
    fee, slip = bundle['fee_bps'],bundle['slippage_bps']
    hold, _ = live_hourly_hold()
    hourly = [t for t in trades if t['tf']=='1h' and t['mode'] in
              ('trend','strong_trend','impulse_speed','impulse','alignment')]
    result = {'contract':'live-buy-replay-contract-loss-audit-v1',
        'scope':'diagnostic_not_release_authorization', 'generated_at':time.time(),
        'bounds':bundle['maximum_available_bounds'], 'days':comparison['paired_days'],
        'population':len(bundle['universe']), 'receipt_sha256':receipt_hash,
        'bundle_sha256':sha((root/'historical_unsigned.json').read_bytes()),
        'audit_source_sha256':sha(Path(__file__).read_bytes()),
        'contract_state':'FAIL' if any(t['state']=='FAIL' for t in tests) else 'UNKNOWN',
        'full_live_poll_parity':'UNKNOWN', 'runtime_eligible':False, 'probes':tests,
        'hourly_base_hold_mismatch':{'n':len(hourly),
            'different_max_hold':sum(t['max_hold_bars']!=hold for t in hourly),
            'held_longer_than_live_base':sum(t['bars_held']>hold for t in hourly)},
        'all_trades':summarize(trades,fee,slip),
        'groups':{k:summarize(v,fee,slip) for k,v in sorted(groups.items())},
        'recorded_portfolios':{arm:r['portfolio'] for arm,r in comparison['reports'].items()},
        'recorded_portfolio_delta_pp':comparison['portfolio_delta_pp'],
        'limitations':['Reconstructed unsigned history, not actual live account performance.',
            'Model exposure overlaps history; no full-window sealed holdout.',
            'Conditional probes do not quantify historical occurrence or causal PnL impact.',
            'Live report forecasts, fetch horizons, discovery cadence, portfolio groups and order unverified.',
            'Cost drag is a fixed-trade cash-account counterfactual, not a rule-ablation result.']}
    if verify(root)[2] != receipt_hash: raise ValueError('receipt drift during audit')
    with output.open('xb') as handle: handle.write(canonical(result))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    kernel=ctypes.windll.kernel32
    kernel.GetCurrentProcess.restype=ctypes.c_void_p
    kernel.SetPriorityClass.argtypes=[ctypes.c_void_p,ctypes.c_uint32]
    kernel.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
    process=kernel.GetCurrentProcess()
    if not kernel.SetPriorityClass(process,0x4000) or not kernel.SetProcessAffinityMask(process,2):
        raise SystemExit('CPU/priority limit unavailable; audit not started')
    result=asyncio.run(audit(args.models,args.output))
    print(json.dumps({k:result[k] for k in ('contract_state','full_live_poll_parity','all_trades','hourly_base_hold_mismatch')}))
