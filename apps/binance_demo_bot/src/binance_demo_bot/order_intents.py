"""P3-A local intent/reservation journal. No broker or execution authority."""
from copy import deepcopy
from decimal import Decimal, InvalidOperation, localcontext
import hashlib
import json
from pathlib import Path
import re

from .adapter import AdapterError
from .interaction_audit import AuditJournal, _sha, sanitize
from .journal import WriterBusy


FIELDS = frozenset(('arm_id', 'decision_id', 'action_seq', 'symbol', 'base_asset',
    'quote_asset', 'side', 'quantity', 'limit_price', 'fee_reserve', 'take_profit',
    'stop_loss', 'asof_ms', 'expires_at_ms', 'market_snapshot_id', 'account_snapshot_id',
    'policy_version', 'model_id', 'source_sha256', 'risk_contract_id'))
REASONS = frozenset(('OMS_INVALID', 'OMS_INTENT_CONFLICT', 'OMS_INSUFFICIENT_ALLOCATION',
    'OMS_FUTURE_SOURCE', 'OMS_EXPIRED', 'OMS_STALE_VERSION', 'OMS_STALE_EPOCH',
    'OMS_ILLEGAL_TRANSITION', 'OMS_ENTRY_KILLED', 'OMS_CAPACITY', 'OMS_NOT_FOUND'))
METHODS = frozenset(('RESERVE', 'SEND_STARTED', 'QUERY_REQUIRED', 'CANCELED', 'KILL'))
MAX_EVENTS = 10000


def _fail(reason='OMS_INVALID'):
    raise AdapterError(reason)


def _identifier(value):
    if not isinstance(value, str) or re.fullmatch(r'[A-Za-z0-9_.:-]{1,128}', value) is None:
        _fail()
    return value


def _time(value):
    if type(value) is not int or value < 0 or value > 2**63-1:
        _fail()
    return value


def _text(value):
    result = format(value, 'f')
    if '.' in result:
        result = result.rstrip('0').rstrip('.')
    return '0' if value == 0 else result


def _amount(value, *, zero=False):
    if not isinstance(value, (str, Decimal)) or isinstance(value, str) and len(value) > 96:
        _fail()
    try:
        result = Decimal(value)
        if (not result.is_finite() or result < 0 or (not zero and result == 0)
                or len(result.as_tuple().digits) > 28 or abs(result.as_tuple().exponent) > 18):
            _fail()
        return _text(result)
    except (InvalidOperation, ValueError, TypeError):
        _fail()


def _allocations(value):
    if not isinstance(value, dict) or not 1 <= len(value) <= 100:
        _fail()
    result = {}
    for arm, assets in value.items():
        _identifier(arm)
        if not isinstance(assets, dict) or not 1 <= len(assets) <= 100:
            _fail()
        result[arm] = {}
        for asset, amount in assets.items():
            if not isinstance(asset, str) or re.fullmatch('[A-Z0-9]{1,20}', asset) is None:
                _fail()
            result[arm][asset] = _amount(amount, zero=True)
    if sanitize(result) != result:
        _fail()
    return result


def _intent(value):
    if not isinstance(value, dict) or set(value) != FIELDS:
        _fail()
    result = deepcopy(value)
    for key in ('arm_id', 'decision_id', 'market_snapshot_id', 'account_snapshot_id',
                'policy_version', 'risk_contract_id'):
        _identifier(result[key])
    if result['model_id'] is not None:
        _identifier(result['model_id'])
    if (not isinstance(result['source_sha256'], str)
            or re.fullmatch('[0-9a-f]{64}', result['source_sha256']) is None):
        _fail()
    _time(result['action_seq'])
    for key in ('asof_ms', 'expires_at_ms'):
        _time(result[key])
    base, quote = result['base_asset'], result['quote_asset']
    if (not isinstance(base, str) or re.fullmatch('[A-Z0-9]{1,20}', base) is None
            or base == 'USDT' or quote != 'USDT' or result['symbol'] != base + quote
            or result['side'] not in ('BUY', 'SELL')):
        _fail()
    for key in ('quantity', 'limit_price', 'fee_reserve'):
        result[key] = _amount(result[key], zero=key == 'fee_reserve')
    if result['side'] == 'BUY':
        for key in ('take_profit', 'stop_loss'):
            result[key] = _amount(result[key])
        if not Decimal(result['stop_loss']) < Decimal(result['limit_price']) < Decimal(result['take_profit']):
            _fail()
    elif result['take_profit'] is not None or result['stop_loss'] is not None:
        _fail()
    if sanitize(result) != result:
        _fail()
    return result


def _reservation(value):
    with localcontext() as context:
        context.prec = 96
        quantity, price, fee = (Decimal(value[k]) for k in ('quantity', 'limit_price', 'fee_reserve'))
        return dict(asset='USDT' if value['side'] == 'BUY' else value['base_asset'],
                    amount=_text(quantity * price + fee if value['side'] == 'BUY' else quantity + fee))


class OrderIntentJournal:
    """Immutable local outbox intents; supplied seed allocations are uncertified."""

    def __init__(self, paths, *, account_alias, episode_id, scope_id, allocations,
                 max_intents=1000, fault_hook=None):
        if type(max_intents) is not int or not 1 <= max_intents <= 1000:
            _fail()
        allocation = _allocations(allocations)
        self._states, self._genesis, self._epoch, self._killed, self._events_n = {}, None, 0, False, 0
        self.max_intents = max_intents
        self._source = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self._journal = None
        try:
            self._journal = AuditJournal(paths, account_alias=account_alias, episode_id=episode_id,
                scope_id=scope_id, database='oms-intents.sqlite', max_bytes=64*1024*1024,
                max_inflight=4, fault_hook=fault_hook)
            with self._journal._lock:
                count = self._journal._con.execute("SELECT COUNT(*) FROM records WHERE kind='DECISION'").fetchone()[0]
                if count > MAX_EVENTS:
                    _fail('OMS_CAPACITY')
                for kind, raw in self._journal._con.execute('SELECT kind,payload_json FROM records ORDER BY seq'):
                    if kind == 'DECISION':
                        try:
                            self._apply(self._plan(json.loads(raw)['data']))
                        except (AdapterError, KeyError, TypeError, ValueError, ArithmeticError):
                            _fail('OMS_INTEGRITY')
                    elif kind not in ('RUN_BEGIN', 'RUN_END'):
                        _fail('OMS_INTEGRITY')
                if self._genesis is None:
                    # Only this opening RUN_BEGIN may exist before genesis.
                    # A crashed/foreign history needs explicit recovery, not new seed funds.
                    if self._journal._last_seq != 1:
                        _fail('OMS_INTEGRITY')
                    self._append('GENESIS', allocations=allocation)
                elif self._genesis != allocation:
                    _fail('OMS_IDENTITY')
                if len(self._states) > max_intents:
                    _fail('OMS_CAPACITY')
                self._append('EPOCH', epoch=self._epoch+1)
                for key in list(self._states):
                    state = self._states[key]
                    if state['state'] == 'SEND_STARTED':
                        self._transition(key, 'QUERY_REQUIRED', expected_version=state['state_version'],
                            lease_epoch=self._epoch, reason='PROCESS_REOPEN')
        except BaseException as error:
            self.close()
            if isinstance(error, WriterBusy):
                raise AdapterError('OMS_BUSY') from None
            if isinstance(error, AdapterError):
                code = {'AUDIT_IDENTITY_MISMATCH': 'OMS_IDENTITY',
                        'AUDIT_INTEGRITY_FAILED': 'OMS_INTEGRITY'}.get(error.reason, error.reason)
                raise AdapterError(code) from None
            raise

    @property
    def lease_epoch(self):
        return self._epoch

    def _ready(self):
        if self._journal is None or self._journal._closed or self._journal._state != 'READY':
            _fail('OMS_UNAVAILABLE')

    def _id(self, value):
        return _sha(dict(identity=self._journal.identity, arm_id=value['arm_id'],
                         decision_id=value['decision_id'], action_seq=value['action_seq']))[:32]

    def _initial(self, value):
        key = self._id(value)
        return dict(intent_id=key, client_order_id='bd_'+key, outbox_id=key, intent=value,
                    reservation=_reservation(value), state='RESERVED', state_version=1,
                    last_reason='LOCAL_RESERVATION', execution_authorized=False)

    @staticmethod
    def _budgets(states, genesis):
        with localcontext() as context:
            context.prec = 96
            result = {arm: {asset: dict(allocated=amount, reserved='0', available=amount)
                           for asset, amount in assets.items()} for arm, assets in genesis.items()}
            for state in states.values():
                if state['state'] == 'CANCELED':
                    continue
                value, reserve = state['intent'], state['reservation']
                budget = result.get(value['arm_id'], {}).get(reserve['asset'])
                if budget is None:
                    _fail('OMS_INSUFFICIENT_ALLOCATION')
                amount = Decimal(budget['reserved']) + Decimal(reserve['amount'])
                if amount > Decimal(budget['allocated']):
                    _fail('OMS_INSUFFICIENT_ALLOCATION')
                budget.update(reserved=_text(amount), available=_text(Decimal(budget['allocated'])-amount))
            return result

    def _plan(self, data):
        """Pure replay validation before append; apply only after durable ACK."""
        base = {'scope', 'operation', 'oms_source_sha256'}
        if (not isinstance(data, dict) or data.get('scope') != 'OMS_INTENTS'
                or not isinstance(data.get('oms_source_sha256'), str)
                or re.fullmatch('[0-9a-f]{64}', data['oms_source_sha256']) is None):
            _fail('OMS_INTEGRITY')
        if self._events_n >= MAX_EVENTS:
            _fail('OMS_CAPACITY')
        states, genesis, epoch, killed = deepcopy(self._states), deepcopy(self._genesis), self._epoch, self._killed
        op = data.get('operation')
        if 'epoch' in data and type(data['epoch']) is not int:
            _fail('OMS_INTEGRITY')
        if op == 'GENESIS':
            if set(data) != base | {'allocations'} or genesis is not None or self._events_n:
                _fail('OMS_INTEGRITY')
            genesis = _allocations(data['allocations'])
            if genesis != data['allocations']:
                _fail('OMS_INTEGRITY')
        elif genesis is None:
            _fail('OMS_INTEGRITY')
        elif op == 'EPOCH':
            if set(data) != base | {'epoch'} or type(data['epoch']) is not int or data['epoch'] != epoch+1:
                _fail('OMS_INTEGRITY')
            epoch = data['epoch']
        elif op == 'BLOCK':
            if (set(data) != base | {'method', 'reason', 'intent_id'} or data['reason'] not in REASONS
                    or data['method'] not in METHODS or epoch < 1):
                _fail('OMS_INTEGRITY')
            if data['intent_id'] is not None and (not isinstance(data['intent_id'], str)
                    or re.fullmatch('[0-9a-f]{32}', data['intent_id']) is None):
                _fail('OMS_INTEGRITY')
        elif op == 'RETRY':
            if set(data) != base | {'intent_id', 'state_version'} or epoch < 1:
                _fail('OMS_INTEGRITY')
            old = states.get(data['intent_id']) if isinstance(data['intent_id'], str) else None
            if (old is None or type(data['state_version']) is not int
                    or data['state_version'] != old['state_version']):
                _fail('OMS_INTEGRITY')
        elif op == 'KILL':
            if set(data) != base | {'reason', 'epoch'} or data['epoch'] != epoch or epoch < 1:
                _fail('OMS_INTEGRITY')
            _identifier(data['reason'])
            killed = True
        elif op == 'RESERVE':
            if set(data) != base | {'state', 'now_ms', 'epoch'} or data['epoch'] != epoch or epoch < 1:
                _fail('OMS_INTEGRITY')
            value = _intent(data['state']['intent'])
            state = self._initial(value)
            if _sha(data['state']) != _sha(state) or state['intent_id'] in states:
                _fail('OMS_INTEGRITY')
            if value['arm_id'] not in genesis:
                _fail()
            now = _time(data['now_ms'])
            if value['asof_ms'] > now:
                _fail('OMS_FUTURE_SOURCE')
            if now >= value['expires_at_ms']:
                _fail('OMS_EXPIRED')
            if killed and value['side'] == 'BUY':
                _fail('OMS_ENTRY_KILLED')
            states[state['intent_id']] = state
        elif op in ('SEND_STARTED', 'QUERY_REQUIRED', 'CANCELED'):
            if set(data) != base | {'state', 'epoch', 'expected_version', 'now_ms'} or data['epoch'] != epoch or epoch < 1:
                _fail('OMS_INTEGRITY')
            new = data['state']
            old = states.get(new['intent_id'])
            if old is None:
                _fail('OMS_NOT_FOUND')
            if type(data['expected_version']) is not int or old['state_version'] != data['expected_version']:
                _fail('OMS_STALE_VERSION')
            required = 'SEND_STARTED' if op == 'QUERY_REQUIRED' else 'RESERVED'
            if old['state'] != required:
                _fail('OMS_ILLEGAL_TRANSITION')
            _identifier(new['last_reason'])
            expected = dict(old, state=op, state_version=old['state_version']+1, last_reason=new['last_reason'])
            if _sha(new) != _sha(expected):
                _fail('OMS_INTEGRITY')
            if op == 'SEND_STARTED':
                now = _time(data['now_ms'])
                if now < old['intent']['asof_ms']:
                    _fail('OMS_FUTURE_SOURCE')
                if now >= old['intent']['expires_at_ms']:
                    _fail('OMS_EXPIRED')
                if killed and old['intent']['side'] == 'BUY':
                    _fail('OMS_ENTRY_KILLED')
                if new['last_reason'] != 'LOCAL_ATTEMPT_ANNOTATION':
                    _fail('OMS_INTEGRITY')
            elif data['now_ms'] is not None:
                _fail('OMS_INTEGRITY')
            states[new['intent_id']] = new
        else:
            _fail('OMS_INTEGRITY')
        self._budgets(states, genesis)
        return states, genesis, epoch, killed, self._events_n+1

    def _apply(self, plan):
        self._states, self._genesis, self._epoch, self._killed, self._events_n = plan

    def _append(self, operation, **payload):
        self._ready()
        data = dict(scope='OMS_INTENTS', operation=operation, oms_source_sha256=self._source, **payload)
        if sanitize(data) != data:
            _fail()
        plan = self._plan(data)
        self._journal.append('DECISION', data)
        self._apply(plan)

    def _block(self, method, reason, intent_id=None):
        if reason not in REASONS:
            _fail(reason)
        if not isinstance(intent_id, str) or re.fullmatch('[0-9a-f]{32}', intent_id) is None:
            intent_id = None
        self._append('BLOCK', method=method, reason=reason, intent_id=intent_id)
        _fail(reason)

    def reserve(self, intent, *, now_ms):
        with self._journal._lock:
            self._ready()
            try:
                _time(now_ms)
                value = _intent(intent)
                if value['arm_id'] not in self._genesis:
                    _fail()
                state = self._initial(value)
                old = self._states.get(state['intent_id'])
                if old is not None:
                    if old['intent'] != value:
                        _fail('OMS_INTENT_CONFLICT')
                    self._append('RETRY', intent_id=old['intent_id'], state_version=old['state_version'])
                    return deepcopy(old)
                if len(self._states) >= self.max_intents:
                    _fail('OMS_CAPACITY')
                self._append('RESERVE', state=state, epoch=self._epoch, now_ms=now_ms)
                return deepcopy(state)
            except AdapterError as error:
                self._block('RESERVE', error.reason)

    def _transition(self, intent_id, operation, *, expected_version, lease_epoch, reason, now_ms=None):
        self._ready()
        if type(lease_epoch) is not int or lease_epoch != self._epoch:
            _fail('OMS_STALE_EPOCH')
        old = self._states.get(intent_id) if isinstance(intent_id, str) else None
        if old is None:
            _fail('OMS_NOT_FOUND')
        if type(expected_version) is not int or expected_version != old['state_version']:
            _fail('OMS_STALE_VERSION')
        _identifier(reason)
        new = dict(old, state=operation, state_version=old['state_version']+1, last_reason=reason)
        self._append(operation, state=new, epoch=self._epoch, expected_version=expected_version, now_ms=now_ms)
        return deepcopy(new)

    def _observed_transition(self, intent_id, operation, **kwargs):
        with self._journal._lock:
            self._ready()
            try:
                return self._transition(intent_id, operation, **kwargs)
            except AdapterError as error:
                self._block(operation, error.reason, intent_id)

    def annotate_send_started(self, intent_id, *, expected_version, lease_epoch, now_ms):
        return self._observed_transition(intent_id, 'SEND_STARTED', expected_version=expected_version,
            lease_epoch=lease_epoch, now_ms=now_ms, reason='LOCAL_ATTEMPT_ANNOTATION')

    def mark_unknown(self, intent_id, *, expected_version, lease_epoch, reason):
        return self._observed_transition(intent_id, 'QUERY_REQUIRED', expected_version=expected_version,
            lease_epoch=lease_epoch, reason=reason)

    def cancel_unsent(self, intent_id, *, expected_version, lease_epoch, reason):
        return self._observed_transition(intent_id, 'CANCELED', expected_version=expected_version,
            lease_epoch=lease_epoch, reason=reason)

    def kill_entries(self, reason):
        with self._journal._lock:
            self._ready()
            try:
                _identifier(reason)
                self._append('KILL', reason=reason, epoch=self._epoch)
            except AdapterError as error:
                self._block('KILL', error.reason)

    def get(self, intent_id):
        with self._journal._lock:
            self._ready()
            if not isinstance(intent_id, str) or intent_id not in self._states:
                _fail('OMS_NOT_FOUND')
            return deepcopy(self._states[intent_id])

    def summary(self):
        with self._journal._lock:
            self._ready()
            manifest = self._journal.manifest()
            unresolved = sum(s['state'] in ('SEND_STARTED', 'QUERY_REQUIRED') for s in self._states.values())
            count = len(self._states)
            return dict(authority='LOCAL_JOURNAL_ONLY', execution_authorized=False, entry_killed=self._killed,
                lease_epoch=self._epoch, intents_N=count, unresolved_n=unresolved,
                unresolved_ratio=(dict(n=unresolved, N=count) if count else None),
                budgets=self._budgets(self._states, self._genesis),
                prefix_last_seq=manifest['last_seq'], prefix_chain_tip=manifest['chain_tip'])

    def close(self):
        if self._journal is not None:
            self._journal.close()
