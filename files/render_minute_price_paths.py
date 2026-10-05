"""Plot genuine frozen regression prices alongside the exact same later facts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from minute_direction_data import sha
from minute_price_paths import METHODS, HORIZONS
import plot_minute_direction_prices as oldplots


def build_price_view(base, pred, widths):
    row = np.flatnonzero((pred['symbol'] == base['symbol']) & (pred['time'] == base['origin_ms']))
    if len(row) != 1: raise ValueError('Price forecast origin is not shared')
    i = row[0]; p0 = base['origin_price']
    if not np.isclose(pred['origin_price'][i], p0, rtol=1e-12): raise ValueError('Origin price mismatch')
    view = {k: base[k] for k in ('symbol', 'origin_ms', 'origin_price', 'clock_ms', 'price')}
    view['target_ms'] = [int(base['origin_ms'] + h * 60_000) for h in (0,) + HORIZONS]
    view['forecasts'] = {}
    for name in METHODS:
        r = pred[name][i]; q = np.asarray(widths[name])
        if r.shape != (5,) or q.shape != (5,) or not np.isfinite(r).all() or not np.isfinite(q).all() or (q < 0).any():
            raise ValueError('Invalid frozen price displacement or interval')
        view['forecasts'][name] = dict(
            price=np.r_[p0, p0 * np.exp(r)].tolist(),
            lower=np.r_[p0, p0 * np.exp(r - q)].tolist(),
            upper=np.r_[p0, p0 * np.exp(r + q)].tolist(), log_return=r.tolist())
    values = dict(zip(base['clock_ms'], base['price']))
    view['actual_endpoints'] = [values[t] for t in view['target_ms']]
    # All methods share the same price axis; observed future affects display only.
    all_prices = [p for p in base['price'] if p is not None]
    for f in view['forecasts'].values(): all_prices.extend(f['lower'] + f['upper'])
    lo, hi = min(all_prices), max(all_prices); pad = max((hi - lo) * .08, p0 * .0001)
    view['price_range'] = [lo - pad, hi + pad]
    return view


def static_plot(view, method, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import pandas as pd
    x = pd.to_datetime(view['clock_ms'], unit='ms', utc=True)
    past = np.array(view['clock_ms']) <= view['origin_ms']; future = np.array(view['clock_ms']) >= view['origin_ms']
    y = np.array([np.nan if v is None else v for v in view['price']])
    tx = pd.to_datetime(view['target_ms'], unit='ms', utc=True); f = view['forecasts'][method]
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.plot(x[past], y[past], color='#1f77b4', label='Observed history before issuance')
    ax.plot(x[future], y[future], color='#f28e2b', label='Actual price AFTER issuance')
    ax.fill_between(tx, f['lower'], f['upper'], color='#8054c3', alpha=.13, label='Calibration-only marginal 90% intervals')
    ax.plot(tx, f['price'], 'o--', color='#8054c3', lw=2, label='Frozen predicted price: +1, +2, +3, +4, +5 min')
    ax.axvline(tx[0], color='black', ls=':', label='One forecast origin')
    ax.set_ylim(*view['price_range']); ax.set_xlim(x[0], x[-1])
    ax.set_title(f"{view['symbol']} | {method} regression | {tx[0]:%Y-%m-%d %H:%M:%S} UTC\n"
        'Five price predictions fixed at one origin; connecting segments are visualization')
    ax.set_ylabel('Mid-price, USDT'); ax.set_xlabel('UTC'); ax.grid(alpha=.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=tx.tz))
    ax.legend(loc='best', fontsize=8); fig.tight_layout(); fig.savefig(output, dpi=150); plt.close(fig)


def html_document(payload):
    from plotly.offline import get_plotlyjs
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace('</', '<\\/')
    page = '''<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>Прогнозная цена — факт</title>
<style>body{font:15px system-ui;max-width:1450px;margin:25px auto;padding:0 20px}header{position:sticky;top:0;background:white;z-index:10;padding:10px}
select{padding:8px;margin:8px}table{border-collapse:collapse;display:block;overflow:auto}td,th{padding:8px;border:1px solid #ddd}.hint{color:#555}</style></head><body>
<h1>Прогнозный график цены: CatBoost / DeepLOB / Ridge</h1>
<p><b>Фиолетовый пунктир с точками — настоящие регрессионные прогнозы цены</b> на +1,+2,+3,+4,+5 минут из одной отсечки.
Синий — доступная история, оранжевый — фактическое движение после выдачи, вертикальный пунктир — момент выдачи.</p>
<p>Точки — выходы моделей. Соединяющие отрезки служат визуализации; промежуточные тики не предсказываются.
Полоса — отдельные 90% интервалы по calibration-остаткам, без гарантии покрытия всей траектории.</p>
<p class="hint">Binance USD-M perpetual mid-price, март–апрель 2026. Это новая регрессия на уже раскрытом историческом TEST,
не независимая forward-проверка. Ни одна модель не получает будущую цену. Примеры общие и выбраны по прошлой доступности,
без отбора по качеству прогноза. Близкий к горизонтальному прогноз сохранён как есть.</p>
<header><label>Монета <select id="asset"></select></label><label>Отсечка UTC <select id="origin"></select></label>
<label><input type="checkbox" id="interval"> 90% интервалы</label>
<label><input type="checkbox" id="focus"> Увеличить прогнозный участок</label></header>
<div id="chart"></div><h2>Прогноз и факт</h2><table id="facts"></table><h2>Сравнение по всему общему TEST</h2>
<p>MAE/RMSE в б.п. относятся к прогнозу изменения цены, а не к близости абсолютных ценовых уровней.
Контроль zero return используется только в метриках. Выбор лучшей модели по этой таблице — описательный.</p>
METRICS
<p>Full Truth Harness: FAIL TH-11 (хеш канонического portfolio replay). Торговая прибыльность этим экспериментом не оценивается.</p>
<p><a href="price_paths.json">Данные графиков и хеши</a> · <a href="metrics_pooled.csv">Метрики CSV</a></p>
<script>LIBRARY</script><script>const payload=PAYLOAD;
const methods=payload.methods, asset=document.getElementById('asset'), origin=document.getElementById('origin');
const iso=t=>new Date(t).toISOString();payload.symbols.forEach(s=>asset.add(new Option(s,s)));
payload.origins_ms.forEach(t=>origin.add(new Option(iso(t),String(t))));
function draw(){
 const v=payload.views[asset.value][origin.value], x=v.clock_ms.map(iso), tx=v.target_ms.map(iso);
 const history=v.price.map((p,i)=>v.clock_ms[i]<=v.origin_ms?p:null), actual=v.price.map((p,i)=>v.clock_ms[i]>=v.origin_ms?p:null);
 const traces=[], shapes=[], annotations=[], intervals=document.getElementById('interval').checked, focus=document.getElementById('focus').checked;
 const shown=v.price.filter((p,i)=>p!==null && (!focus || v.clock_ms[i]>=v.origin_ms));
 methods.forEach(m=>{const f=v.forecasts[m];shown.push(...f.price);if(intervals)shown.push(...f.lower,...f.upper);});
 const lo=Math.min(...shown),hi=Math.max(...shown),pad=Math.max((hi-lo)*.08,v.origin_price*.0001),priceRange=[lo-pad,hi+pad];
 const layout={height:1250,margin:{l:95,r:35,t:65,b:60},showlegend:true,hovermode:'x unified',shapes,annotations};
 methods.forEach((method,i)=>{
  const suffix=i===0?'':String(i+1), xa='x'+suffix, ya='y'+suffix, top=1-i/3, f=v.forecasts[method];
  layout['xaxis'+suffix]={domain:[0,1],anchor:ya,type:'date',range:[focus?iso(v.origin_ms):x[0],x[x.length-1]],matches:i?'x':undefined,title:{text:'UTC'}};
  layout['yaxis'+suffix]={domain:[top-.27,top-.035],anchor:xa,title:{text:'Цена, USDT'},range:priceRange};
  annotations.push({xref:'paper',yref:'paper',x:0,y:top,text:method+' regression | одна отсечка, пять прогнозных цен',showarrow:false,xanchor:'left',font:{size:17}});
  shapes.push({type:'line',xref:xa,yref:ya+' domain',x0:iso(v.origin_ms),x1:iso(v.origin_ms),y0:0,y1:1,line:{color:'black',dash:'dot'}});
  traces.push({x,y:history,type:'scatter',mode:'lines',line:{color:'#1f77b4'},name:'История до выдачи',showlegend:i===0,legendgroup:'history',connectgaps:false,xaxis:xa,yaxis:ya});
  if(intervals){
   traces.push({x:tx,y:f.lower,type:'scatter',mode:'lines',line:{width:0},showlegend:false,hoverinfo:'skip',xaxis:xa,yaxis:ya});
   traces.push({x:tx,y:f.upper,type:'scatter',mode:'lines',line:{width:0},fill:'tonexty',fillcolor:'rgba(128,84,195,.13)',
     name:'90% интервалы (marginal)',showlegend:i===0,hoverinfo:'skip',legendgroup:'interval',xaxis:xa,yaxis:ya});
  }
  traces.push({x,y:actual,type:'scatter',mode:'lines',line:{color:'#f28e2b'},name:'Факт после выдачи',showlegend:i===0,legendgroup:'actual',connectgaps:false,xaxis:xa,yaxis:ya});
  traces.push({x:tx,y:f.price,type:'scatter',mode:'lines+markers',line:{color:'#8054c3',dash:'dash',width:2},marker:{size:8},
    name:'ПРОГНОЗ цены +1..+5 мин',showlegend:i===0,legendgroup:'forecast',xaxis:xa,yaxis:ya,hovertemplate:'Прогноз %{y:.4f} USDT<extra>'+method+'</extra>'});
 });
 Plotly.react('chart',traces,layout,{responsive:true,displaylogo:false});
 let text='<tr><th>Модель</th><th>Горизонт</th><th>Прогноз, USDT</th><th>Факт, USDT</th><th>Ошибка, USDT</th><th>Прогноз изменения</th></tr>';
 methods.forEach(m=>v.forecasts[m].price.slice(1).forEach((p,j)=>{const a=v.actual_endpoints[j+1];text+='<tr><td>'+m+'</td><td>+'+(j+1)+
  ' мин</td><td>'+p.toFixed(4)+'</td><td>'+(a===null?'UNKNOWN':a.toFixed(4))+'</td><td>'+(a===null?'UNKNOWN':(p-a).toFixed(4))+
  '</td><td>'+((p/v.origin_price-1)*100).toFixed(4)+'%</td></tr>';}));document.getElementById('facts').innerHTML=text;
}
asset.onchange=draw;origin.onchange=draw;document.getElementById('interval').onchange=draw;document.getElementById('focus').onchange=draw;draw();</script></body></html>'''
    return page.replace('METRICS', payload['metrics_html']).replace('LIBRARY', get_plotlyjs()).replace('PAYLOAD', data)


def render(folder, classifier, books_dir):
    import pandas as pd
    result = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
    receipt = json.loads((folder / 'verification.json').read_text(encoding='utf-8'))
    if result['status'] != 'COMPLETED_RETROSPECTIVE_PRICE_PATHS' or receipt['status'] != 'PASS':
        raise ValueError('Price run not completed and natively verified')
    if sha(folder / 'result.json') != receipt['result_sha256'] or sha(folder / 'predictions.npz') != receipt['predictions_sha256']:
        raise ValueError('Frozen price evidence changed')
    old, old_receipt, classifier_pred, books = oldplots.load_inputs(classifier, books_dir)
    if result['registration']['classifier_hashes'] != dict(result=old_receipt['result_sha256'], predictions=old_receipt['predictions_sha256']):
        raise ValueError('Classifier anchor experiment changed')
    with np.load(folder / 'predictions.npz', allow_pickle=False) as f: pred = {k: f[k] for k in f.files}
    origins = oldplots.select_origins(classifier_pred, books, old['cuts'])
    rows = [dict(method=name, **s) for name in METHODS for s in result['metrics'][name]['pooled']]
    assetrows = [dict(method=name, asset=s, **v) for name in METHODS for s, scores in result['metrics'][name]['assets'].items() for v in scores]
    table = pd.DataFrame(rows); table.to_csv(folder / 'metrics_pooled.csv', index=False)
    pd.DataFrame(assetrows).to_csv(folder / 'metrics_assets.csv', index=False)
    daily = []
    for name in METHODS:
        for j, h in enumerate(HORIZONS):
            mask = pred['scored']; errors = (pred[name][mask, j] - pred['actual_returns'][mask, j]) * 10000
            groups = pd.DataFrame({'day': pred['time'][mask] // 86400000, 'absolute_error': np.abs(errors), 'square_error': errors ** 2})
            for day, group in groups.groupby('day'):
                daily.append(dict(method=name, horizon_min=h, date=pd.to_datetime(day * 86400000, unit='ms', utc=True).isoformat()[:10],
                    n=len(group), mae_bp=float(group.absolute_error.mean()), rmse_bp=float(np.sqrt(group.square_error.mean()))))
    pd.DataFrame(daily).to_csv(folder / 'daily_metrics.csv', index=False)
    table['direction_pct'] = table.direction_correct / table.direction_n * 100
    table['interval_coverage_pct'] = table.interval_covered / table.interval_n * 100
    visible = ['method', 'horizon_min', 'n', 'mae_bp', 'rmse_bp', 'zero_return_mae_bp', 'direction_correct', 'direction_n', 'direction_pct',
        'observed_majority_correct', 'interval_covered', 'interval_n', 'interval_coverage_pct']
    payload = dict(methods=list(METHODS), symbols=list(books), origins_ms=origins.tolist(),
        result_sha256=receipt['result_sha256'], predictions_sha256=receipt['predictions_sha256'],
        classifier_hashes=result['registration']['classifier_hashes'],
        book_sha256={s: meta['sha256'] for s, meta in result['coverage']['assets'].items()},
        metrics_html=table[visible].to_html(index=False, float_format=lambda v: f'{v:.4f}'), views={})
    images = folder / 'plots'; images.mkdir(exist_ok=True)
    for symbol, book in books.items():
        payload['views'][symbol] = {}
        for t in origins:
            base = oldplots.build_view(classifier_pred, symbol, t, book)
            payload['views'][symbol][str(t)] = build_price_view(base, pred, result['calibration_widths'])
        for name in METHODS:
            static_plot(payload['views'][symbol][str(origins[0])], name, images / f'{symbol}_{name}.png')
    (folder / 'price_paths.json').write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    (folder / 'price_paths.html').write_text(html_document(payload), encoding='utf-8')
    print(table[visible].to_string(index=False)); print(folder / 'price_paths.html')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('folder', type=Path)
    parser.add_argument('--classifier', type=Path, required=True); parser.add_argument('--books', type=Path, required=True)
    a = parser.parse_args(); render(a.folder, a.classifier, a.books)
