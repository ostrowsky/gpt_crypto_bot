"""Bounded owner-thread delivery attribution; raw evidence is never health output."""
from collections import deque
from copy import deepcopy
from decimal import Context, Decimal, localcontext
import hashlib
import base64
import json
from .adapter import AdapterError


class WirePacket(dict):
    """Dictionary semantics unchanged; exact public wire retained only in flight."""
    def __init__(self, payload, raw):
        super().__init__(payload)
        self.raw_wire = raw


def _duration(end, start):
    if end is None or start is None or end < start: return None
    with localcontext(Context(prec=50)): return Decimal(end-start)/1_000_000


class DeliveryMetrics:
    FIELDS = ("collection_ms", "reader_wait_ms", "prepare_ms", "writer_wait_ms",
              "begin_ms", "insert_ms", "commit_ms", "end_to_end_ms")

    def __init__(self, window):
        if type(window) is not int or not 1 <= window <= 4096:
            raise AdapterError("INVALID_PARAMETERS")
        self.recent = deque(maxlen=window)
        self.attempted = self.durable = self.passed = self.failed = self.late = 0
        self.evidence_failures = 0
        self.evidence_state = "NOT_OBSERVED"

    def note(self, frames, pending, collected, acquired, finished, reason, source_id, session_id):
        timing = pending.timing if pending else {}
        first = min(frame[2] for frame in frames)
        group = dict(frames_N=len(frames), source_snapshot_id=source_id, session_id=session_id,
            commit_state=timing.get("state", "NOT_STARTED"), failure_reason=reason,
            collection_ms=_duration(collected, first), reader_wait_ms=_duration(acquired, collected),
            prepare_ms=_duration(timing.get("prepare_end_ns"), timing.get("prepare_start_ns")),
            writer_wait_ms=_duration(timing.get("writer_acquired_ns"), timing.get("writer_wait_started_ns")),
            begin_ms=_duration(timing.get("begin_finished_ns"), timing.get("writer_acquired_ns")),
            insert_ms=_duration(timing.get("insert_finished_ns"), timing.get("begin_finished_ns")),
            commit_ms=_duration(timing.get("commit_finished_ns"), timing.get("insert_finished_ns")),
            end_to_end_ms=_duration(finished, first))
        self.attempted += 1
        self.durable += group["commit_state"]=="COMMITTED"
        self.passed += reason is None
        self.failed += reason is not None
        self.late += reason=="WS_PROCESSING_LAG" or group["end_to_end_ms"] is None or group["end_to_end_ms"]>250
        self.recent.append(group)
        return deepcopy(group)

    def health(self):
        maximum = {name:max((group[name] for group in self.recent if group[name] is not None), default=None)
                   for name in self.FIELDS}
        return dict(scope="READ_ONCE_STEADY_STATE", startup_observed=False,
            attempted_groups_n=self.attempted, durable_groups_n=self.durable,
            passed_groups_n=self.passed, failed_groups_n=self.failed, late_groups_n=self.late,
            recent_groups_n=len(self.recent), window_limit_N=self.recent.maxlen,
            last_group=deepcopy(self.recent[-1]) if self.recent else None, max_ms=maximum,
            evidence_failures_n=self.evidence_failures, last_evidence_state=self.evidence_state)


def lag_evidence(frames, pending, group):
    committed = pending is not None and pending.timing["state"]=="COMMITTED"
    records = []
    for payload, wall, mono in frames:
        raw = getattr(payload, "raw_wire", None)
        decoded = json.dumps(payload,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False)
        try: text = raw.decode("utf-8") if raw is not None else None
        except UnicodeError: text = None
        records.append(dict(received_at_ms=wall, received_monotonic_ns=mono,
            decoded_json=decoded, canonical_decoded_sha256=hashlib.sha256(decoded.encode()).hexdigest(),
            raw_utf8=text, wire_base64=base64.b64encode(raw).decode("ascii") if raw is not None else None,
            wire_sha256=hashlib.sha256(raw).hexdigest() if raw is not None else None,
            exact_wire_observed=raw is not None))
    return dict(operation="MARKET_GROUP_LAG_EVIDENCE", reason="WS_PROCESSING_LAG",
        frames_N=len(records), frames=records, timing=group, committed=committed,
        committed_snapshot_ids=list(pending.values) if committed else [],
        order_send_enabled=False, production_qualified=False)
