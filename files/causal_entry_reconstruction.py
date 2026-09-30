"""Immutable retrospective price diagnostic; never an actual fill or live policy."""
from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

import aiohttp
import independent_signal_evaluator as evaluator
import policy_provenance as provenance
from historical_signal_evaluation import freeze, sha

CONTRACT = 'next-minute-open-fixed-t5-v1'
MINUTE = 60_000


def canonical(rows):
    groups = defaultdict(list)
    for row in rows:
        if not isinstance(row, dict) or not row.get('id'):
            raise ValueError('unknown observation identity')
        groups[row['id']].append(row)
    result, conflicts, removed = [], 0, 0
    for variants in groups.values():
        row = dict(variants[0])
        removed += len(variants)-1
        # No outcome-based choice or unsafe merging of observation variants.
        if any(v != variants[0] for v in variants[1:]):
            conflicts += 1
            row['reconstruction_integrity'] = 'conflicting_duplicate'
        result.append(row)
    return result, {'input_rows': len(rows), 'unique_ids': len(result),
                    'removed_duplicate_rows': removed, 'quarantined_ids': conflicts}


def valid_bar(bar, interval):
    if not isinstance(bar, list) or len(bar) < 7:
        raise ValueError('missing raw bar')
    ts = int(bar[0])
    o, h, l, c, v = map(float, bar[1:6])
    if (ts % interval or int(bar[6]) != ts+interval-1
            or not all(math.isfinite(x) for x in (o, h, l, c, v))
            or min(o, h, l, c) <= 0 or v < 0
            or h < max(o, l, c) or l > min(o, h, c)):
        raise ValueError('invalid OHLCV')
    return ts, o, c


def deduplicate_live(dataset, run):
    """Archive before atomic cleanup. Conflicting IDs never enter training."""
    import critic_dataset
    if dataset.resolve() != critic_dataset.CRITIC_FILE.resolve():
        raise ValueError('cleanup must use the configured locked dataset')
    run.mkdir(parents=True, exist_ok=False)
    with critic_dataset._dataset_io_lock():
        original = dataset.read_bytes()
        freeze(run/'original.jsonl', original)
        rows, counts = canonical([json.loads(l) for l in original.splitlines() if l.strip()])
        quarantine = [r['id'] for r in rows if r.get('reconstruction_integrity')]
        clean = [r for r in rows if not r.get('reconstruction_integrity')]
        raw = b''.join(json.dumps(r, ensure_ascii=False, allow_nan=False).encode()+b'\n' for r in clean)
        freeze(run/'clean.jsonl', raw)
        report = {**counts, 'retained_rows': len(clean), 'quarantined_ids': quarantine,
                  'original_sha256': sha(run/'original.jsonl'), 'clean_sha256': sha(run/'clean.jsonl'),
                  'generated_at': provenance.utc_iso(), 'trading_policy_changed': False}
        freeze(run/'manifest.json', json.dumps(report, sort_keys=True).encode())
        if counts['removed_duplicate_rows']:
            tmp = dataset.with_name(dataset.name+'.dedup.tmp')
            freeze(tmp, raw)
            critic_dataset._atomic_replace_with_retry(tmp, dataset)
        if sha(dataset) != report['clean_sha256'] and counts['removed_duplicate_rows']:
            raise ValueError('live cleanup integrity mismatch')
    return report


def execution_return(row, now):
    """Recompute entirely from raw evidence. No rounded-return inversion."""
    if row.get('reconstruction_integrity'):
        raise ValueError('quarantined duplicate')
    evidence = row.get('execution_reconstruction') or {}
    if evidence.get('contract') != CONTRACT:
        raise ValueError('unknown execution contract')
    decision = provenance.parse_utc((row.get('decision_provenance') or {}).get('decision_time'))
    if decision is None:
        raise ValueError('unknown decision')
    decision_ms = decision.timestamp()*1000
    next_open = (math.floor(decision_ms/MINUTE)+1)*MINUTE
    minute_bars = evidence.get('minute_bars') or []
    if len(minute_bars) != 3:
        raise ValueError('missing minute proof')
    values = [valid_bar(b, MINUTE) for b in minute_bars]
    if [v[0] for v in values] != [next_open-2*MINUTE, next_open, next_open+MINUTE]:
        raise ValueError('wrong minute boundary')
    interval = int(provenance.timeframe_delta(row['tf']).total_seconds()*1000)
    window = evidence.get('target_bars') or []
    target_values = [valid_bar(b, interval) for b in window]
    bar_ts = int(row['bar_ts'])
    if (not provenance.observation_provenance_valid(row)
            or provenance.parse_utc(row['provenance']['feature_time']) != provenance.feature_cutoff(bar_ts, row['tf'])):
        raise ValueError('feature/decision boundary mismatch')
    if [v[0] for v in target_values] != [bar_ts+i*interval for i in range(7)]:
        raise ValueError('incomplete T5 horizon')
    target_ms = bar_ts+6*interval
    recorded = provenance.parse_utc(evidence.get('retrieved_at'))
    if (not decision_ms < next_open < target_ms or recorded is None or recorded > now
            or recorded.timestamp()*1000 < max(target_ms, next_open+2*MINUTE)
            or any(not re.fullmatch('[0-9a-f]{64}', str(evidence.get(k, '')))
                   for k in ('minute_response_sha256', 'target_response_sha256'))):
        raise ValueError('causal timing/source mismatch')
    original = float((row.get('labels') or {}).get('ret_5'))
    expected = (target_values[5][2]/target_values[0][2]-1)*100
    if not math.isfinite(original) or abs(original-expected) > 0.00011:
        raise ValueError('original target disagrees with exchange history')
    return (target_values[5][2]/values[1][1]-1)*100


def verify(run):
    """Bind every reconstructed raw bar to a preserved exchange response."""
    receipt = json.loads((run/'receipt.json').read_bytes())
    expected_files = {'result.json', 'repaired.jsonl', 'original.jsonl', 'repair_manifest.json',
                      'manifest.json', 'candidate.json', 'responses_manifest.json'}
    if set(receipt) != expected_files:
        raise ValueError('unknown receipt scope')
    for name, expected in receipt.items():
        if sha(run/name) != expected:
            raise ValueError('reconstruction artifact hash mismatch: '+name)
    responses = json.loads((run/'responses_manifest.json').read_bytes())
    result = json.loads((run/'result.json').read_bytes())
    by_hash = defaultdict(list)
    for item in responses['requests']:
        path = run/'responses'/item['path']
        if (path.parent.resolve() != (run/'responses').resolve()
                or sha(path) != item['sha256']):
            raise ValueError('exchange response hash mismatch')
        by_hash[item['sha256']].append((item, json.loads(path.read_bytes())))
    for line in (run/'repaired.jsonl').read_bytes().splitlines():
        row = json.loads(line)
        evidence = row.get('execution_reconstruction')
        if not evidence:
            continue
        for prefix, bars, tf in (('minute', evidence['minute_bars'], '1m'),
                                 ('target', evidence['target_bars'], row['tf'])):
            options = by_hash[evidence[prefix+'_response_sha256']]
            if not any(item['key'][:2] == [row['sym'], tf] and all(b in raw for b in bars)
                       for item, raw in options):
                raise ValueError('reconstructed bar not bound to symbol/timeframe response')
        value = execution_return(row, provenance.parse_utc(result['generated_at']))
        if abs(value-evidence.get('diagnostic_ret5', value)) > 1e-10:
            raise ValueError('stored execution return disagrees with raw prices')
    manifest = json.loads((run/'repair_manifest.json').read_bytes())
    if (manifest['source_dataset_sha256'] != sha(run/'original.jsonl')
            or manifest['candidate_sha256'] != sha(run/'candidate.json')
            or result['candidate_sha256'] != manifest['candidate_sha256']
            or result['dataset_sha256'] != sha(run/'repaired.jsonl')
            or result.get('runtime_eligible') is not False):
        raise ValueError('repair/result input binding changed')
    return result


async def reconstruct(dataset, registry, run, now=None, responses_from=None):
    now = now or datetime.now(timezone.utc)
    run.mkdir(parents=True, exist_ok=False)
    import critic_dataset
    from contextlib import nullcontext
    barrier = critic_dataset._dataset_io_lock() if dataset.resolve() == critic_dataset.CRITIC_FILE.resolve() else nullcontext()
    with barrier:
        with dataset.open('rb') as source, (run/'original.jsonl').open('xb') as target:
            shutil.copyfileobj(source, target)
    freeze(run/'candidate.json', (registry/'candidate.json').read_bytes())
    freeze(run/'manifest.json', (registry/'manifest.json').read_bytes())
    source_names = ('causal_entry_reconstruction.py', 'independent_signal_evaluator.py', 'critic_dataset.py',
                    'ml_candidate_ranker.py', 'ml_signal_model.py', 'policy_provenance.py', 'config.py')
    freeze(run/'repair_manifest.json', json.dumps({
        'source_dataset_sha256': sha(run/'original.jsonl'),
        'candidate_sha256': sha(run/'candidate.json'), 'contract': CONTRACT,
        'created_at': provenance.utc_iso(now),
        'sources': {n: sha(Path(__file__).with_name(n)) for n in source_names},
        'scope': 'entire_available_file; exposure embargo only; no symbol filter',
        'runtime_eligible': False,
    }, sort_keys=True).encode())
    rows, counts = canonical([json.loads(l) for l in (run/'original.jsonl').read_bytes().splitlines() if l.strip()])
    model = json.loads((run/'candidate.json').read_bytes())
    scopes = model['evaluation_provenance']['split_scopes']
    exposure = [provenance.parse_utc(s[k]) for s in scopes.values()
                for k in ('last_feature_time', 'last_label_time', 'last_label_recorded_at')]
    if any(t is None for t in exposure):
        raise ValueError('unknown exposure')
    start = max(exposure)+evaluator.timedelta(hours=evaluator.EMBARGO_HOURS)
    requests, jobs = {}, {}
    for row in rows:
        feature = provenance.parse_utc((row.get('provenance') or {}).get('feature_time'))
        decision = provenance.parse_utc((row.get('decision_provenance') or {}).get('decision_time'))
        if (feature is None or not start <= feature <= now or row.get('reconstruction_integrity')
                or not provenance.observation_provenance_valid(row) or decision is None
                or row.get('tf') not in {'15m', '1h', '4h'}
                or not re.fullmatch('[A-Z0-9]+USDT', str(row.get('sym', '')))
                or not provenance.label_provenance_valid(row, 'ret_5')
                or (row.get('labels') or {}).get('ret_5') is None):
            continue
        interval = int(provenance.timeframe_delta(row['tf']).total_seconds()*1000)
        bar_ts = int(row['bar_ts'])
        next_open = (math.floor(decision.timestamp()*1000/MINUTE)+1)*MINUTE
        if next_open >= bar_ts+6*interval or now.timestamp()*1000 < bar_ts+7*interval:
            continue
        # Hour buckets batch nearby decisions, without using later OHLC for entry.
        minute_key = (row['sym'], '1m', next_open//3_600_000)
        target_key = (row['sym'], row['tf'], bar_ts//86_400_000)
        for key, lo, hi in ((minute_key, next_open-2*MINUTE, next_open+MINUTE),
                            (target_key, bar_ts, bar_ts+6*interval)):
            old = requests.setdefault(key, [lo, hi])
            old[0], old[1] = min(old[0], lo), max(old[1], hi)
        jobs[row['id']] = (minute_key, target_key, next_open, interval)
    semaphore = asyncio.Semaphore(4)
    responses, failures = {}, []
    cached = {}
    if responses_from:
        verify(responses_from)
        old = json.loads((responses_from/'responses_manifest.json').read_bytes())
        cached = {tuple(item['key']): item for item in old['requests']}
    import config
    async with aiohttp.ClientSession() as session:
        async def fetch(key, bounds):
            async with semaphore:
                params = {'symbol': key[0], 'interval': key[1], 'startTime': bounds[0],
                          'endTime': bounds[1], 'limit': 1000}
                try:
                    prior = cached.get(key)
                    if prior and prior['bounds'] == bounds:
                        raw = (responses_from/'responses'/prior['path']).read_bytes()
                        retrieved = prior['retrieved_at']
                    else:
                        async with session.get(config.BINANCE_REST+'/api/v3/klines', params=params,
                                               timeout=aiohttp.ClientTimeout(total=30)) as response:
                            response.raise_for_status()
                            raw = await response.read()
                        retrieved = provenance.utc_iso()
                    bars = json.loads(raw)
                    if not isinstance(bars, list) or len(bars) >= 1000:
                        raise ValueError('truncated or malformed response')
                    step = MINUTE if key[1] == '1m' else int(provenance.timeframe_delta(key[1]).total_seconds()*1000)
                    mapped = {}
                    for b in bars:
                        ts, _, _ = valid_bar(b, step)
                        if ts in mapped or not bounds[0] <= ts <= bounds[1]:
                            raise ValueError('duplicate or outside request')
                        mapped[ts] = b
                    digest = hashlib.sha256(raw).hexdigest()
                    filename = f'{key[0]}_{key[1]}_{key[2]}.json'
                    freeze(run/'responses'/filename, raw)
                    responses[key] = (mapped, digest, retrieved, filename)
                except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, TypeError, OSError) as exc:
                    failures.append({'key': list(key), 'error': str(exc)})
        tasks = [fetch(k, b) for k, b in requests.items()]
        # Bounded batches make progress observable and avoid a giant task queue.
        for offset in range(0, len(tasks), 100):
            await asyncio.gather(*tasks[offset:offset+100])
            print(json.dumps({'requests_completed': min(offset+100, len(tasks)),
                              'requests_total': len(tasks), 'failures': len(failures)}), flush=True)
    counts['price_requests'] = len(requests)
    counts['price_request_failures'] = len(failures)
    counts['restored_rows'] = 0
    reasons = defaultdict(int)
    ended = datetime.now(timezone.utc)
    for row in rows:
        job = jobs.get(row['id'])
        if job is None:
            continue
        mk, tk, nxt, interval = job
        try:
            minute, mh, mt, _ = responses[mk]
            target, th, tt, _ = responses[tk]
            row['execution_reconstruction'] = {
                'contract': CONTRACT, 'exact_decision_quote': None,
                'minute_bars': [minute[t] for t in (nxt-2*MINUTE, nxt, nxt+MINUTE)],
                'target_bars': [target[int(row['bar_ts'])+i*interval] for i in range(7)],
                'minute_response_sha256': mh, 'target_response_sha256': th,
                'retrieved_at': max(mt, tt),
                'decision_reference_price': float(minute[nxt-2*MINUTE][4]),
                'simulated_entry_price': float(minute[nxt][1]),
                'entry_time': provenance.utc_iso(datetime.fromtimestamp(nxt/1000, timezone.utc)),
            }
            row['execution_reconstruction']['diagnostic_ret5'] = execution_return(row, ended)
            counts['restored_rows'] += 1
        except (ValueError, KeyError, TypeError) as exc:
            row.pop('execution_reconstruction', None)
            reasons['missing_market_response_or_bar' if isinstance(exc, KeyError) else str(exc)] += 1
    freeze(run/'repaired.jsonl', b''.join(json.dumps(r, ensure_ascii=False, allow_nan=False).encode()+b'\n' for r in rows))
    result = evaluator.evaluate(run, run/'repaired.jsonl', now=ended, historical=True, execution=True)
    by_id = {r['id']: r for r in rows}
    old_daily = defaultdict(list)
    for pair in result['pairs']:
        baseline = float(by_id[pair['baseline_id']]['labels']['ret_5'])
        challenger = float(by_id[pair['candidate_id']]['labels']['ret_5'])
        old_daily[pair['group'][:10]].append(challenger-baseline)
    old_values = [sum(v)/len(v) for v in old_daily.values()]
    result['price_correction_same_pairs'] = {
        'paired_groups': len(result['pairs']), 'valid_days': len(old_values),
        'old_closed_bar_mean_daily_delta_pp': sum(old_values)/len(old_values) if old_values else None,
        'new_next_minute_mean_daily_delta_pp': result['learning_quality']['mean_daily_ret5_delta_pp'],
    }
    result['repair'] = {**counts, 'restoration_failures': dict(reasons),
                        'exact_historical_execution_price': 'UNKNOWN',
                        'source_dataset_sha256': sha(run/'original.jsonl')}
    result['limitations'].append('recorded historical decision timestamps cannot retroactively certify score immutability or original quote capture')
    freeze(run/'result.json', json.dumps(result, sort_keys=True, allow_nan=False).encode())
    freeze(run/'responses_manifest.json', json.dumps({
        'requests': [{'key': list(k), 'bounds': requests[k], 'sha256': v[1], 'path': v[3], 'retrieved_at': v[2]}
                     for k, v in responses.items()], 'failures': failures,
    }, sort_keys=True).encode())
    freeze(run/'receipt.json', json.dumps({p: sha(run/p) for p in
         ('result.json', 'repaired.jsonl', 'original.jsonl', 'repair_manifest.json',
          'manifest.json', 'candidate.json', 'responses_manifest.json')}, sort_keys=True).encode())
    verify(run)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--registry', type=Path, default=evaluator.REGISTRY)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--responses-from', type=Path, help='Reuse hash-verified public responses; never reuse conclusions')
    parser.add_argument('--deduplicate-live', action='store_true', help='Archive and atomically clean the configured training stream')
    args = parser.parse_args()
    if args.deduplicate_live:
        report = deduplicate_live(args.dataset, args.run)
        print(json.dumps({k: v for k, v in report.items() if k != 'quarantined_ids'}))
        print(json.dumps({'quarantined_id_count': len(report['quarantined_ids'])}))
    else:
        report = asyncio.run(reconstruct(args.dataset, args.registry, args.run, responses_from=args.responses_from))
        print(json.dumps({'repair': report['repair'], 'quality': report['learning_quality']}))
