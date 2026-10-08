"""Bounded recovery for known transient I/O failures; no dataset lock deletion."""
RETRY_LIMIT=3
RETRY_SECONDS=300


def retryable(error):
    seen=set();current=error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current,TimeoutError) and str(current).startswith('timeout acquiring critic_dataset lock:'):
            return True
        if isinstance(current,PermissionError) and getattr(current,'winerror',None) in (32,33):
            return True
        current=current.__cause__ or current.__context__
    return False
