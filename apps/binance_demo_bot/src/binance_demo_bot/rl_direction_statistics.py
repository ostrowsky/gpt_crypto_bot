"""RDE11 calendar-block descriptive uncertainty, no promotion authority."""
from datetime import date
import math
import random


def paired_block_intervals(candidate_daily, reference_daily, *, draws=5000, seed=42,
                           confidence=1-.05/6, block_days=7):
    if set(candidate_daily) != set(reference_daily):
        raise ValueError('UNPAIRED_CALENDAR_DAYS')
    if (type(draws) is not int or draws <= 0 or type(seed) is not int
            or type(block_days) is not int or block_days <= 0
            or not math.isfinite(confidence) or not 0 < confidence < 1):
        raise ValueError('INVALID_SAMPLING_PROTOCOL')
    keys = sorted(candidate_daily)
    dates = [date.fromisoformat(k) for k in keys]
    if any(d.isoformat() != k for d,k in zip(dates,keys)):
        raise ValueError('INVALID_CALENDAR_DAY')
    if any((b-a).days != 1 for a,b in zip(dates,dates[1:])):
        raise ValueError('NON_CONTIGUOUS_CALENDAR_DAYS')
    values, differences = [], []
    for key in keys:
        a,b = candidate_daily[key],reference_daily[key]
        if any(type(v) not in (float,int) or not math.isfinite(v) for v in (a,b)):
            raise ValueError('UNKNOWN_DAILY_VALUE')
        values.append(a); differences.append(a-b)
    n = len(keys)
    if n >= 30 and block_days > n:
        raise ValueError('INVALID_SAMPLING_PROTOCOL')
    result = dict(paired_n=n,delta_mean=sum(differences)/n if n else None,
        absolute_mean=sum(values)/n if n else None,delta_lower=None,delta_upper=None,
        absolute_lower=None,absolute_upper=None,confidence=confidence,draws=draws,
        seed=seed,block_days=block_days,promotion_allowed=False,
        state='INCONCLUSIVE' if n < 30 else 'DESCRIPTIVE_REUSED_HOLDOUT')
    if n < 30:
        return result
    # Circular moving blocks, same sampled dates for paired and absolute means.
    full, tail = divmod(n,block_days)
    def sums(source,length):
        extended=source+source[:length]
        prefix=[0.]
        for v in extended: prefix.append(prefix[-1]+v)
        return [prefix[i+length]-prefix[i] for i in range(n)]
    av,dv=sums(values,block_days),sums(differences,block_days)
    at,dt=sums(values,tail),sums(differences,tail)
    absolute,delta=[],[];rng=random.Random(seed)
    for _ in range(draws):
        indices=[rng.randrange(n) for _ in range(full)]
        last=rng.randrange(n) if tail else 0
        absolute.append((sum(av[i] for i in indices)+at[last])/n)
        delta.append((sum(dv[i] for i in indices)+dt[last])/n)
    def quantile(source,p):
        source.sort(); x=(len(source)-1)*p; i=int(x)
        return source[i]+(source[min(i+1,len(source)-1)]-source[i])*(x-i)
    alpha=(1-confidence)/2
    result.update(delta_lower=quantile(delta,alpha),delta_upper=quantile(delta,1-alpha),
        absolute_lower=quantile(absolute,alpha),absolute_upper=quantile(absolute,1-alpha))
    return result
