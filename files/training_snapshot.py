"""Content-addressed evaluator snapshots; readers never block dataset replacement."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import uuid
from validated_ranker_rollout import canonical

CONTRACT = 'immutable-pre-holdout-training-v1'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def pointer(output):
    return output.with_suffix('.snapshot.json')


def publish(tmp, output, rows, cutoff):
    checksum = digest(tmp)
    folder = output.parent/'training_snapshots'
    folder.mkdir(parents=True, exist_ok=True)
    target = folder/(checksum+'.jsonl')
    if target.exists():
        if digest(target) != checksum:
            raise ValueError('existing immutable training snapshot corrupt')
        tmp.unlink()
    else:
        tmp.rename(target)
    body = {'contract': CONTRACT, 'path': str(target.relative_to(output.parent)),
            'sha256': checksum, 'bytes': target.stat().st_size, 'rows': rows,
            'cutoff': cutoff.isoformat(), 'scope': 'pre-holdout-only-not-learning-approval'}
    destination = pointer(output)
    staging = destination.with_name(destination.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with staging.open('xb') as handle:
            handle.write(canonical(body)); handle.flush(); os.fsync(handle.fileno())
        staging.replace(destination)
    finally:
        staging.unlink(missing_ok=True)
    return body


def resolve(output):
    body = json.loads(pointer(output).read_bytes())
    if body.get('contract') != CONTRACT:
        raise ValueError('invalid training snapshot contract')
    checksum = body['sha256']
    if (not isinstance(checksum, str) or len(checksum) != 64
            or any(c not in '0123456789abcdef' for c in checksum)):
        raise ValueError('invalid snapshot digest')
    expected = Path('training_snapshots')/(checksum+'.jsonl')
    if Path(body['path']) != expected:
        raise ValueError('training snapshot path mismatch')
    path = (output.parent/expected).resolve()
    if not path.is_relative_to(output.parent.resolve()):
        raise ValueError('snapshot escapes protected intake')
    if (path.stat().st_size != body['bytes'] or digest(path) != checksum
            or not isinstance(body['rows'], int) or isinstance(body['rows'], bool) or body['rows'] < 0):
        raise ValueError('training snapshot content mismatch')
    return path, body
