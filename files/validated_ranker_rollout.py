"""Disabled-by-default bounded model overlay. Authority keys are never auto-created."""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import time
from process_lock import process_lock

ROOT = Path(__file__).resolve().parents[1] / '.runtime/validated_ranker_rollout'
CONTRACT = 'validated-ranker-rollout-v1'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def seal(body, key):
    if not key or len(key) < 32:
        raise ValueError('authority key must contain at least 32 bytes')
    return {'body': body, 'mac': hmac.new(key, canonical(body), hashlib.sha256).hexdigest()}


def unseal(value, key):
    expected = seal(value['body'], key)['mac']
    if not hmac.compare_digest(expected, value['mac']):
        raise ValueError('authority signature mismatch')
    return value['body']


def atomic_pointer(root, value, expected):
    root.mkdir(parents=True, exist_ok=True)
    with process_lock(root / 'pointer.lock'):
        path = root / 'active.json'
        current = sha(path.read_bytes()) if path.exists() else None
        if current != expected:
            raise ValueError('release pointer changed concurrently')
        tmp = root / 'active.tmp'
        with tmp.open('wb') as handle:
            handle.write(canonical(value))
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(path)


def rollout(root, candidate_raw, authorization, key, champion_raw, expected=None, now=None):
    now = time.time() if now is None else now
    ticket = validate_ticket(authorization, key, champion_raw, now)
    if sha(candidate_raw) != ticket['candidate_sha256']:
        raise ValueError('candidate model mismatch')
    json.loads(candidate_raw)
    root.mkdir(parents=True, exist_ok=True)
    model = root / (ticket['candidate_sha256'] + '.json')
    if model.exists():
        if model.read_bytes() != candidate_raw:
            raise ValueError('immutable model changed')
    else:
        with model.open('xb') as handle:
            handle.write(candidate_raw)
    atomic_pointer(root, authorization, expected)


def validate_ticket(value, key, champion_raw, now):
    ticket = unseal(value, key)
    from independent_portfolio_gate import source_hash
    if (ticket['contract'] != CONTRACT or ticket['stage'] not in ('CANARY', 'PROMOTED')
            or not ticket['issued_at'] <= now < ticket['expires_at']
            or ticket['expires_at']-ticket['issued_at'] > 86400
            or ticket['champion_sha256'] != sha(champion_raw)
            or ticket['evaluator_sha256'] != source_hash()
            or ticket['max_bonus'] != 1.0
            or ticket['fraction'] != (0.05 if ticket['stage'] == 'CANARY' else 1.0)):
        raise ValueError('stale, mismatched or unsafe release ticket')
    return ticket


def assigned(symbol, ticket):
    bucket = int(sha((ticket['candidate_sha256']+'|'+symbol.upper()).encode())[:8], 16)
    return bucket / 2**32 < ticket['fraction']


def select(root, champion_raw, symbol, key, enabled=False, now=None):
    """Any failure falls back to caller's champion; never mutates it."""
    if not enabled:
        return None
    try:
        ticket = validate_ticket(json.loads((root/'active.json').read_bytes()), key,
                                 champion_raw, time.time() if now is None else now)
        if not assigned(symbol, ticket):
            return None
        raw = (root/(ticket['candidate_sha256']+'.json')).read_bytes()
        if sha(raw) != ticket['candidate_sha256']:
            return None
        return json.loads(raw), ticket
    except (ValueError, OSError, TypeError, KeyError, OverflowError):
        return None


def bounded_bonus(value):
    return max(-1.0, min(1.0, float(value))) if math.isfinite(float(value)) else 0.0


def rollback(root, expected):
    atomic_pointer(root, {'state': 'ROLLED_BACK'}, expected)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--expected', default=None, help='expected SHA of existing pointer; omitted only for first install')
    p.add_argument('--rollback', action='store_true')
    p.add_argument('--authorization', type=Path)
    p.add_argument('--candidate', type=Path)
    p.add_argument('--champion', type=Path)
    args = p.parse_args()
    if args.rollback:
        rollback(args.root, args.expected)
    else:
        if not all((args.authorization, args.candidate, args.champion)):
            p.error('activation requires authorization, candidate and champion')
        rollout(args.root, args.candidate.read_bytes(), json.loads(args.authorization.read_bytes()),
                os.environ.get('RANKER_EVALUATOR_KEY', '').encode(), args.champion.read_bytes(), args.expected)
