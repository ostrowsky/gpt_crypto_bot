"""Maximum-local-hourly BTC lead hypothesis; diagnostic, never release authority."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median

BAR = 3600000
TARGETS = ('SOLUSDT', 'ETHUSDT')
THRESHOLDS = (0.25, 0.5, 1.0)
HORIZONS = (1, 3, 6, 12, 24)
EXITS = ('fixed24', 'btc_nonpositive', 'btc_drop', 'target_nonpositive')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def merge_rows(groups):
    result = {}
    for rows in groups:
        for r in rows:
            if isinstance(r, dict):
                t = int(r['t'])
                values = tuple(float(r[k]) for k in ('o','h','l','c'))
                close_time = t + BAR - 1  # recovered serialized archive, not raw certification
            else:
                t = int(r[0])
                values = tuple(float(r[k]) for k in (1, 2, 3, 4))
                close_time = int(r[6])
            if t % BAR or close_time != t + BAR - 1:
                raise ValueError('not closed hourly candle')
            if any(not math.isfinite(v) or v <= 0 for v in values):
                raise ValueError('invalid OHLC')
            if t in result and result[t] != values:
                raise ValueError(f'conflicting raw OHLC at {t}')
            result[t] = values
    return result


def pct(a, b):
    return (a / b - 1) * 100


def summarize(values):
    return {'n': len(values), 'positive': sum(v > 0 for v in values),
            'nonpositive': sum(v <= 0 for v in values),
            'mean_pct': mean(values) if values else None,
            'median_pct': median(values) if values else None}


def valid_window(data, t, before=1, after=25):
    return all(t + k * BAR in rows for rows in data.values()
               for k in range(-before, after + 1))


def events(data, start, end, threshold):
    out, next_allowed = [], start
    btc = data['BTCUSDT']
    for t in sorted(btc):
        if t < start or t < next_allowed or t + 25 * BAR >= end:
            continue
        if not valid_window(data, t):
            continue
        if pct(btc[t][3], btc[t-BAR][3]) >= threshold:
            out.append(t)
            next_allowed = t + 25 * BAR
    return out


def net_return(entry, exit_price):
    # Explicit fees/slippage on both causal next-open fills.
    return pct(exit_price * (1-0.0005) * (1-0.00075),
               entry * (1+0.0005) * (1+0.00075))


def trade(data, symbol, signal, exit_rule):
    entry_t = signal + BAR
    exit_t = entry_t + 24 * BAR
    for t in range(entry_t, exit_t, BAR):
        btc_r = pct(data['BTCUSDT'][t][3], data['BTCUSDT'][t-BAR][3])
        target_r = pct(data[symbol][t][3], data[symbol][t-BAR][3])
        if ((exit_rule == 'btc_nonpositive' and btc_r <= 0)
                or (exit_rule == 'btc_drop' and btc_r <= -0.25)
                or (exit_rule == 'target_nonpositive' and target_r <= 0)):
            exit_t = t + BAR
            break
    entry, exit_price = data[symbol][entry_t][0], data[symbol][exit_t][0]
    return {'signal_ms': signal, 'entry_ms': entry_t, 'exit_ms': exit_t,
            'entry': entry, 'exit': exit_price,
            'net_pct': net_return(entry, exit_price)}


def basket(data, stamps, rule):
    rows = [{s: trade(data, s, t, rule) for s in TARGETS} for t in stamps]
    returns = [mean(r[s]['net_pct'] for s in TARGETS) for r in rows]
    equity, peak, dd = 1., 1., 0.
    for r in returns:
        equity *= 1 + r / 100
        peak = max(peak, equity)
        dd = max(dd, (1-equity/peak)*100)
    return {**summarize(returns), 'basket_compounded_pct': (equity-1)*100 if rows else None,
            'trade_close_drawdown_pct': dd if rows else None, 'trades': rows}


def choose_threshold(data, start, split):
    scored = [(basket(data, events(data, start, split, q), 'fixed24'), q)
              for q in THRESHOLDS]
    eligible = [(r, q) for r, q in scored if r['n'] >= 20]
    return max(eligible, key=lambda x: (x[0]['basket_compounded_pct'], -x[1]))[1] if eligible else None


def forward_stats(data, stamps):
    result = {}
    btc = data['BTCUSDT']
    for s in TARGETS:
        result[s] = {}
        for h in HORIZONS:
            vals, btcvals, examples = [], [], []
            for t in stamps:
                v = pct(data[s][t+h*BAR][3], data[s][t][3])
                b = pct(btc[t+h*BAR][3], btc[t][3])
                vals.append(v); btcvals.append(b)
                if v <= 0 and len(examples) < 5:
                    examples.append({'signal_ms': t, 'target_pct': v, 'btc_pct': b})
            pos = [(v, b) for v, b in zip(vals, btcvals) if b > 0]
            result[s][str(h)] = {**summarize(vals),
                'btc_positive_denominator': len(pos),
                'target_at_least_2x_btc': sum(v >= 2*b for v, b in pos),
                'median_multiple_when_btc_positive': median(v/b for v,b in pos) if pos else None,
                'counterexamples': examples,
                'scope': 'close-to-close diagnostic; not next-open trading returns'}
    return result


def correlation(pairs):
    if len(pairs) < 20:
        return None
    x, y = zip(*pairs); mx, my = mean(x), mean(y)
    den = math.sqrt(sum((a-mx)**2 for a in x)*sum((b-my)**2 for b in y))
    return sum((a-mx)*(b-my) for a,b in pairs)/den if den else None


def analyze(data):
    start = max(min(rows) for rows in data.values())
    end = min(max(rows) for rows in data.values()) + BAR
    split = start + int((end-start)/BAR*0.7)*BAR
    selected = choose_threshold(data, start, split)
    results = {}
    for phase, a, b in (('train', start, split), ('test', split, end)):
        phase_rows = {}
        for q in THRESHOLDS:
            stamps = events(data, a, b, q)
            phase_rows[str(q)] = {'events': len(stamps), 'forward': forward_stats(data, stamps)}
        baseline_stamps = [t for t in sorted(data['BTCUSDT'])
                          if a <= t and t+25*BAR < b and valid_window(data,t)]
        lag = {}
        for s in TARGETS:
            lag[s] = {}
            for k in (0,1,2,4,8):
                pairs = [(pct(data['BTCUSDT'][t][3],data['BTCUSDT'][t-BAR][3]),
                          pct(data[s][t+k*BAR][3],data[s][t+(k-1)*BAR][3]))
                         for t in baseline_stamps]
                lag[s][str(k)] = {'n': len(pairs), 'correlation': correlation(pairs)}
        stamps = events(data, a, b, selected) if selected is not None else []
        exits = {r: basket(data, stamps, r) for r in EXITS}
        fixed = [mean(t[s]['net_pct'] for s in TARGETS) for t in exits['fixed24']['trades']]
        for r in EXITS:
            vals = [mean(t[s]['net_pct'] for s in TARGETS) for t in exits[r]['trades']]
            exits[r]['paired_delta_vs_fixed24'] = summarize([x-y for x,y in zip(vals,fixed)])
        results[phase] = {'thresholds': phase_rows, 'baseline': forward_stats(data,baseline_stamps),
                          'lag_correlations': lag, 'selected_threshold_exits': exits}
    return {'state': 'DIAGNOSTIC_ONLY', 'runtime_eligible': False,
            'bounds_ms': [start,end], 'split_ms': split, 'selected_train_threshold': selected,
            'rows_per_symbol': {s:len(r) for s,r in data.items()}, 'results': results,
            'limitations': ['hourly ordering cannot prove sub-hour BTC lead',
                'historical cache, not point-in-time live candidate population',
                'recovered archive close times inferred from hourly cadence, not raw certification',
                'overlapping unconditional baseline is descriptive only',
                'no statistical promotion gate or live-bot portfolio alpha',
                'three thresholds/horizons/exits: multiple-testing exploration',
                'no guarantee of future returns; full prospective verification required']}


def run(root, output):
    data, inputs = {}, {}
    for symbol in ('BTCUSDT', *TARGETS):
        paths = sorted(set((root/'.runtime/price_cluster_cache').glob(symbol+'_1h_*.json')) |
                       set((root/'.runtime/closed_grid_policy_replay/20261001_max_archive_v1/market').glob(symbol+'_1h_*.json')))
        paths = [p for p in paths if not p.name.endswith('.manifest.json')]
        if not paths:
            raise ValueError('missing hourly history: '+symbol)
        groups = []
        for p in paths:
            raw = p.read_bytes(); inputs[str(p)] = hashlib.sha256(raw).hexdigest()
            groups.append(json.loads(raw))
        data[symbol] = merge_rows(groups)
    report = analyze(data)
    if any(digest(Path(p)) != sha for p,sha in inputs.items()):
        raise ValueError('input drift')
    report['input_hashes'] = inputs
    report['source_sha256'] = digest(Path(__file__))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as handle:
        json.dump(report,handle,allow_nan=False)
    print(json.dumps({'state':report['state'],'bounds_ms':report['bounds_ms'],
                      'selected_train_threshold':report['selected_train_threshold']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent.parent)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    run(args.root,args.output)
