"""Offline common-origin price/history/fact views of frozen direction forecasts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from minute_direction_data import sha

METHODS = ('Prior', 'Momentum', 'Logistic', 'CatBoost', 'DeepLOB')
HORIZONS = (1, 3, 5)
STEP = 10_000
HISTORY = 20 * 60_000
BAND = .0002
CLASSES = ('DOWN', 'NEUTRAL', 'UP')
COLORS = ('#d62728', '#808080', '#16994b')


def mid_prices(book):
    price = (book['book'][:, 0] + book['book'][:, 2]) / 2
    valid = (book['segment'] >= 0) & np.isfinite(price) & (price > 0)
    return np.where(valid, price, np.nan)


def select_origins(pred, books, cuts):
    """Clock anchors and past prefixes only; future labels are never read."""
    common = None
    for symbol, book in books.items():
        time = book['time']; price = mid_prices(book)
        valid = np.isfinite(price)
        # Count missing observations in each past-only 20-minute prefix.
        prefix = np.r_[0, np.cumsum(~valid)]
        n = HISTORY // STEP
        indices = np.arange(n, len(time))
        good = (prefix[indices + 1] - prefix[indices - n] == 0)
        good &= time[indices] - time[indices - n] == HISTORY
        eligible = time[indices[good]]
        issued = pred['time'][pred['symbol'] == symbol]
        eligible = np.intersect1d(eligible, issued)
        common = eligible if common is None else np.intersect1d(common, eligible)
    common = common[(common >= cuts['test']) & (common <= cuts['end'])]
    if not len(common):
        raise ValueError('No common issued origin with a valid past-only history')
    anchors = np.arange(cuts['test'], cuts['end'] + 1, 6 * 60 * 60_000)
    indices = np.searchsorted(common, anchors)
    return np.unique(common[indices[indices < len(common)]])


def category(log_return):
    return 0 if log_return < -BAND else 2 if log_return > BAND else 1


def build_view(pred, symbol, origin, book):
    matches = np.flatnonzero((pred['symbol'] == symbol) & (pred['time'] == origin))
    if len(matches) != 1:
        raise ValueError('Expected exactly one immutable inference row')
    row = matches[0]
    time = book['time']; price = mid_prices(book)
    index = np.searchsorted(time, origin)
    if index >= len(time) or time[index] != origin or not np.isfinite(price[index]):
        raise ValueError('Issuance mid-price unavailable')
    p0 = float(price[index])
    clock = np.arange(origin - HISTORY, origin + 5 * 60_000 + 1, STEP)
    positions = np.searchsorted(time, clock)
    values = np.full(len(clock), np.nan)
    safe = positions < len(time)
    safe[safe] &= time[positions[safe]] == clock[safe]
    values[safe] = price[positions[safe]]
    forecasts = {}
    for name in METHODS:
        probs = np.asarray(pred[name][row], dtype=float)
        if probs.shape != (3, 3) or not np.isfinite(probs).all() or (probs < 0).any():
            raise ValueError('Invalid frozen probabilities')
        if not np.allclose(probs.sum(axis=1), 1):
            raise ValueError('Probabilities do not sum to one')
        forecasts[name] = [dict(horizon_min=h, target_ms=int(origin + h * 60_000),
            probabilities=p.tolist(), predicted_class=CLASSES[int(np.argmax(p))])
            for h, p in zip(HORIZONS, probs)]
    facts = []
    for h in HORIZONS:
        target = int(origin + h * 60_000)
        value = values[(clock == target).nonzero()[0][0]]
        facts.append(dict(horizon_min=h, target_ms=target,
            price=float(value) if np.isfinite(value) else None,
            actual_class=CLASSES[category(np.log(value / p0))] if np.isfinite(value) else 'UNKNOWN'))
    return dict(symbol=symbol, origin_ms=int(origin), origin_price=p0,
        neutral_price_bounds=[float(p0 * np.exp(-BAND)), float(p0 * np.exp(BAND))],
        clock_ms=clock.tolist(), price=[float(v) if np.isfinite(v) else None for v in values],
        forecasts=forecasts, facts=facts)


def load_inputs(folder, books_dir):
    result = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
    receipt = json.loads((folder / 'verification.json').read_text(encoding='utf-8'))
    if receipt.get('status') != 'PASS':
        raise ValueError('Native verification receipt is not PASS')
    for key, filename in [('result_sha256', 'result.json'), ('predictions_sha256', 'test_predictions.npz')]:
        if receipt.get(key) != sha(folder / filename):
            raise ValueError('Frozen verification hash mismatch: ' + filename)
    if result['status'] != 'COMPLETED_RETROSPECTIVE_L2_DIAGNOSTIC':
        raise ValueError('Incomplete benchmark')
    with np.load(folder / 'test_predictions.npz', allow_pickle=False) as f:
        pred = {k: f[k] for k in f.files}
    books = {}
    for symbol, meta in result['coverage']['assets'].items():
        path = books_dir / (symbol + '.npz')
        if sha(path) != meta['sha256']:
            raise ValueError('Prepared price source drift: ' + symbol)
        with np.load(path, allow_pickle=False) as f:
            books[symbol] = {k: f[k] for k in ('time', 'book', 'segment')}
        if np.any(np.diff(books[symbol]['time']) != STEP):
            raise ValueError('Prepared clock must be a complete ten-second grid')
    return result, receipt, pred, books


def static_plot(view, method, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import pandas as pd

    x = pd.to_datetime(view['clock_ms'], unit='ms', utc=True)
    y = np.array([np.nan if v is None else v for v in view['price']])
    past = np.array(view['clock_ms']) <= view['origin_ms']
    future = np.array(view['clock_ms']) >= view['origin_ms']
    fig, (ax, prob) = plt.subplots(2, 1, figsize=(13, 6), sharex=True,
        gridspec_kw={'height_ratios': [4, 1]})
    ax.plot(x[past], y[past], color='#1f77b4', label='Observed history (available at issuance)')
    ax.plot(x[future], y[future], color='#f28e2b', label='Actual movement AFTER issuance')
    cutoff = pd.to_datetime(view['origin_ms'], unit='ms', utc=True)
    ax.axvline(cutoff, color='black', ls='--', label='Frozen forecast issued here')
    ax.axhspan(*view['neutral_price_bounds'], alpha=.08, color='gray', label='Neutral class +/-2bp from origin')
    for fc, fact in zip(view['forecasts'][method], view['facts']):
        target = pd.to_datetime(fc['target_ms'], unit='ms', utc=True)
        p = fc['probabilities']; klass = CLASSES.index(fc['predicted_class'])
        ax.axvspan(target - pd.Timedelta(seconds=15), target + pd.Timedelta(seconds=15), color=COLORS[klass], alpha=.14)
        ax.text(target, .98, f"+{fc['horizon_min']}m {fc['predicted_class']}\nP={p[klass]:.1%}",
            transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=8, color=COLORS[klass])
        if fact['price'] is not None:
            ax.scatter(target, fact['price'], color='#f28e2b', s=30, zorder=4)
        base = 0
        for k in range(3):
            prob.bar(target, p[k], bottom=base, width=30 / 86400, color=COLORS[k],
                label=CLASSES[k] if fc['horizon_min'] == 1 else None)
            base += p[k]
    ax.set_title(f"{view['symbol']} | {method} | origin {cutoff:%Y-%m-%d %H:%M:%S} UTC\n"
        'Color bands = frozen direction forecast; no predicted price magnitude')
    ax.set_ylabel('Mid-price, USDT'); ax.grid(alpha=.2); ax.legend(loc='lower left', fontsize=8)
    prob.set_ylabel('Probability'); prob.set_ylim(0, 1); prob.legend(loc='upper left', ncol=3, fontsize=8)
    prob.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=cutoff.tz))
    prob.set_xlabel('UTC | forecast endpoints +1, +3, +5 minutes'); prob.grid(axis='y', alpha=.2)
    fig.tight_layout(); fig.savefig(output, dpi=150); plt.close(fig)


def html_document(payload):
    from plotly.offline import get_plotlyjs
    # Single embedded library, no network dependencies or external JSON fetches.
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace('</', '<\\/')
    page = '''<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>История — прогноз — факт</title>
<style>body{font:15px system-ui;margin:25px auto;max-width:1450px;padding:0 20px}select{padding:8px;margin:8px}
header{position:sticky;top:0;background:white;z-index:10;padding:10px;border-bottom:1px solid #ddd}
.hint{color:#555}#facts{border-collapse:collapse}td,th{padding:7px;border:1px solid #ddd}</style></head><body>
<h1>Каждая модель: история цены → замороженный прогноз → факт</h1>
<p>Ретроспективный TEST, март–апрель 2026. Цена — mid-price стакана Binance USD-M perpetual, USDT.</p>
<p>Синий — история, доступная до отсечки. Оранжевый — фактическая цена после прогноза.
Вертикальный пунктир — момент выдачи. Цветные полосы на +1/+3/+5 мин — прогноз направления;
вероятности классов показаны под каждым графиком. Серый ценовой диапазон — нейтральный класс ±2 б.п.
от цены в момент выдачи. Это не интервал неопределённости. Модели не предсказывают величину цены.</p>
<p class="hint">Все модели показывают один период и один момент выдачи. Примеры выбраны по шестичасовым
часовым отсечкам и доступности прошлой истории, без отбора по будущему результату. Пропуски не соединяются.</p>
<header><label>Монета <select id="asset"></select></label><label>Момент прогноза, UTC <select id="origin"></select></label>
<span id="context"></span></header><div id="chart"></div><h2>Прогноз и факт на точных горизонтах</h2><table id="facts"></table>
<p><a href="comparison.html">Сравнительные метрики всех методов</a> · <a href="price_forecasts.json">Данные графиков и хеши</a></p>
<p class="hint">Full Truth Harness: FAIL TH-11 (несовпадение хеша канонического portfolio replay).
Графики — исследовательская иллюстрация, без разрешения торговать.</p>
<script>LIBRARY</script><script>const payload=PAYLOAD;
const methods=payload.methods, classes=['DOWN','NEUTRAL','UP'], colors=['#d62728','#808080','#16994b'];
const asset=document.getElementById('asset'), origin=document.getElementById('origin');
const iso=t=>new Date(t).toISOString();
payload.symbols.forEach(s=>asset.add(new Option(s,s)));
payload.origins_ms.forEach(t=>origin.add(new Option(iso(t),String(t))));
function draw(){
 const v=payload.views[asset.value][origin.value], cutoff=iso(v.origin_ms), x=v.clock_ms.map(iso);
 const history=v.price.map((p,i)=>v.clock_ms[i]<=v.origin_ms?p:null);
 const actual=v.price.map((p,i)=>v.clock_ms[i]>=v.origin_ms?p:null);
 const traces=[], shapes=[], annotations=[];const layout={height:2250,template:'plotly_white',
 margin:{l:90,r:40,t:55,b:65},showlegend:true,barmode:'stack',hovermode:'x unified',shapes,annotations};
 methods.forEach((method,i)=>{
   const n=i*2+1, suffix=n===1?'':String(n), xa='x'+suffix, ya='y'+suffix;
   const pn=n+1, px='x'+pn, py='y'+pn, top=1-i*.2;
   layout['xaxis'+suffix]={domain:[0,1],anchor:ya,range:[x[0],x[x.length-1]],type:'date',showticklabels:false,
      matches:i?'x':undefined};
   layout['yaxis'+suffix]={domain:[top-.135,top-.02],anchor:xa,title:{text:'Mid-price, USDT'},autorange:true};
   layout['xaxis'+pn]={domain:[0,1],anchor:py,range:[x[0],x[x.length-1]],type:'date',matches:'x',title:{text:'UTC'}};
   layout['yaxis'+pn]={domain:[top-.177,top-.145],anchor:px,range:[0,1],title:{text:'P'},tickvals:[0,1]};
   traces.push({x,y:history,type:'scatter',mode:'lines',line:{color:'#1f77b4'},name:'История до выдачи',
      legendgroup:'history',showlegend:i===0,connectgaps:false,xaxis:xa,yaxis:ya});
   traces.push({x,y:actual,type:'scatter',mode:'lines',line:{color:'#f28e2b'},name:'Факт после выдачи',
      legendgroup:'actual',showlegend:i===0,connectgaps:false,xaxis:xa,yaxis:ya});
   shapes.push({type:'line',xref:xa,yref:ya+' domain',x0:cutoff,x1:cutoff,y0:0,y1:1,line:{color:'black',dash:'dash'}});
   shapes.push({type:'rect',xref:xa,yref:ya,x0:x[0],x1:x[x.length-1],y0:v.neutral_price_bounds[0],
      y1:v.neutral_price_bounds[1],fillcolor:'gray',opacity:.08,line:{width:0},layer:'below'});
   annotations.push({xref:'paper',yref:'paper',x:0,y:top-.005,text:method+' | прогноз направления на 1 / 3 / 5 минут',
      showarrow:false,xanchor:'left',font:{size:17}});
   const forecasts=v.forecasts[method];
   forecasts.forEach(f=>{
      const k=classes.indexOf(f.predicted_class), t=iso(f.target_ms);
      shapes.push({type:'rect',xref:xa,yref:ya+' domain',x0:iso(f.target_ms-15000),x1:iso(f.target_ms+15000),
          y0:0,y1:1,fillcolor:colors[k],opacity:.12,line:{width:0},layer:'below'});
      annotations.push({xref:xa,yref:ya+' domain',x:t,y:1,text:'+'+f.horizon_min+'m '+f.predicted_class+'<br>P='+
          (100*f.probabilities[k]).toFixed(1)+'%',showarrow:false,yanchor:'top',font:{color:colors[k],size:11}});
   });
   for(let k=0;k<3;k++)traces.push({type:'bar',x:forecasts.map(f=>iso(f.target_ms)),y:forecasts.map(f=>f.probabilities[k]),
       width:30000,marker:{color:colors[k]},name:classes[k],legendgroup:classes[k],showlegend:i===0,xaxis:px,yaxis:py,
       hovertemplate:classes[k]+': %{y:.3f}<extra>'+method+'</extra>'});
   traces.push({type:'scatter',mode:'markers',x:v.facts.map(f=>iso(f.target_ms)),y:v.facts.map(f=>f.price),
       text:v.facts.map(f=>'Факт +'+f.horizon_min+'m: '+f.actual_class),marker:{color:'#f28e2b',size:8},
       name:'Факт на горизонте',showlegend:false,xaxis:xa,yaxis:ya,hovertemplate:'%{text}<br>%{y}<extra></extra>'});
 });
 Plotly.react('chart',traces,layout,{responsive:true,displaylogo:false});
 document.getElementById('context').textContent='20 мин истории / 5 мин после выдачи';
 let table='<tr><th>Метод</th><th>Горизонт</th><th>Прогноз</th><th>P DOWN / NEUTRAL / UP</th><th>Факт</th><th>Цена факта, USDT</th></tr>';
 methods.forEach(m=>v.forecasts[m].forEach((f,j)=>{const fact=v.facts[j];table+='<tr><td>'+m+'</td><td>+'+f.horizon_min+
   ' мин</td><td>'+f.predicted_class+'</td><td>'+f.probabilities.map(p=>(100*p).toFixed(1)+'%').join(' / ')+
   '</td><td>'+fact.actual_class+'</td><td>'+(fact.price===null?'UNKNOWN':fact.price.toFixed(4))+'</td></tr>';}));
 document.getElementById('facts').innerHTML=table;
}
asset.onchange=draw;origin.onchange=draw;draw();</script></body></html>'''
    return page.replace('LIBRARY', get_plotlyjs()).replace('PAYLOAD', data)


def render(folder, books_dir):
    result, receipt, pred, books = load_inputs(folder, books_dir)
    origins = select_origins(pred, books, result['cuts'])
    payload = dict(methods=list(METHODS), symbols=list(books), origins_ms=origins.tolist(),
        selection='first common past-eligible issued origin on/after fixed six-hour TEST anchors',
        history_minutes=20, future_minutes=5, neutral_log_return_band=BAND,
        result_sha256=receipt['result_sha256'], predictions_sha256=receipt['predictions_sha256'],
        book_sha256={s: m['sha256'] for s, m in result['coverage']['assets'].items()}, views={})
    images = folder / 'price_forecast_plots'; images.mkdir(exist_ok=True)
    for symbol, book in books.items():
        payload['views'][symbol] = {str(t): build_view(pred, symbol, t, book) for t in origins}
        for method in METHODS:
            static_plot(payload['views'][symbol][str(origins[0])], method, images / f'{symbol}_{method}.png')
    (folder / 'price_forecasts.json').write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    (folder / 'price_forecasts.html').write_text(html_document(payload), encoding='utf-8')
    print(f'{len(origins)} common origins, {len(books) * len(METHODS)} static plots; {folder / "price_forecasts.html"}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--books', type=Path, required=True)
    args = parser.parse_args()
    render(args.folder, args.books)
