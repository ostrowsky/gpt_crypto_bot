"""Reconcile post-close diagnostics, never change executions or create trades."""
import math

FIELDS = ('cooldown_blocked_count', 'cooldown_positive_blocked_count', 'cooldown_harm_pct')


def get(row, key):
    return row[key] if isinstance(row, dict) else getattr(row, key)


def identity(row):
    key = tuple(get(row, k) for k in ('sym', 'tf', 'entry_ts', 'exit_ts'))
    if (not all(isinstance(v, str) and v for v in key[:2])
            or any(type(v) is not int or v <= 0 for v in key[2:])
            or key[3] < key[2]):
        raise ValueError('invalid closed trade identity')
    return key


def extend_index(index, rows):
    additions = {}
    for row in rows:
        key = identity(row)
        if key in index or key in additions:
            raise ValueError('duplicate closed trade identity')
        additions[key] = row
    index.update(additions)


def reconcile(index, state):
    """Apply cumulative diagnostics by durable identity, after close append.

    Only the latest closed trade for each symbol can still receive annotations.
    Validate the whole update set before applying; the index is not a checkpoint.
    """
    updates = []
    for symbol, snapshot in state.get('last_closed_by_symbol', {}).items():
        key = identity(snapshot)
        if symbol != key[0] or key not in index:
            raise ValueError('checkpoint refers to unknown closed trade')
        target = index[key]
        values = {field: get(snapshot, field) for field in FIELDS}
        for field, value in values.items():
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0
                    or (field != FIELDS[2] and type(value) is not int)
                    or value < get(target, field)):
                raise ValueError('invalid or regressing cooldown annotation')
        if values[FIELDS[1]] > values[FIELDS[0]]:
            raise ValueError('positive cooldown count exceeds total')
        updates.append((target, values))
    for target, values in updates:
        if isinstance(target, dict):
            target.update(values)
        else:
            for field, value in values.items():
                setattr(target, field, value)
