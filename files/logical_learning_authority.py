"""Explicit same-user trust boundary; never an OS-access isolation attestation."""
from dataclasses import dataclass
from pathlib import Path
import secrets
from validated_ranker_rollout import unseal

CONTRACT = 'logical-market-policy-certification-v1'


@dataclass(frozen=True)
class LogicalAuthority:
    key: bytes


def provision(base):
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    for name in ('coverage.key', 'evaluator.key'):
        path = base/name
        if not path.exists():
            with path.open('xb') as handle:
                handle.write(secrets.token_bytes(48))
        if path.stat().st_size != 48:
            raise ValueError('invalid local authority key')


def material(deployment):
    if (deployment.get('isolation_mode') != 'logical_same_user' or
            deployment.get('logical_isolation_accepted') is not True):
        raise ValueError('logical trust boundary was not accepted')
    # Fixed relative location, not chosen by an evidence manifest.
    base = Path(deployment['registry']).parent/'authority'
    return LogicalAuthority((base/'coverage.key').read_bytes()), (base/'evaluator.key').read_bytes()


def verify(attestation, authority):
    cert = unseal(attestation, authority.key)
    if (cert.get('contract') != CONTRACT or cert.get('isolation_mode') != 'logical_same_user'
            or cert.get('os_access_isolation') is not False
            or cert.get('training_holdout_excluded') is not True):
        raise ValueError('logical certification cannot claim OS isolation or training leakage')
    return cert
