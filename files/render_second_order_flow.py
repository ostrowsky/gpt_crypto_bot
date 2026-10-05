"""Offline synchronized history / frozen second-scale forecast / later mid-price."""
from __future__ import annotations
import argparse,json,csv
from pathlib import Path
import numpy as np
from minute_direction_data import sha,SYMBOLS
from evaluate_second_order_flow import load_asset
from second_order_flow_models import HORIZONS,REGISTERED

METHODS=('CatBoost_OFI','CatBoost_Book','Ridge_OFI')


def select_origins(pred,cuts):
    """Fixed 12-hour anchors; selection uses only past eligibility, never outcomes."""
    common=None
    for s in SYMBOLS:
        t=pred['time'][pred['symbol']==s]
        common=t if common is None else np.intersect1d(common,t)
    if common is None or not len(common):raise ValueError('No shared issuance origins')
    anchors=np.arange(cuts['test'],cuts['end']-31000,12*3600000,dtype=np.int64)
    result=[]
    for anchor in anchors:
        i=np.searchsorted(common,anchor)
        if i<len(common):result.append(int(common[i]))
    return sorted(set(result))


def build_view(data,pred,widths,symbol,origin):
    row=np.flatnonzero((pred['symbol']==symbol)&(pred['time']==origin))
    if len(row)!=1:raise ValueError('Forecast origin is not unique/shared')
    row=int(row[0]);p0=float(pred['origin_price'][row]);clock=np.arange(origin-60000,origin+30001,1000)
    idx=np.searchsorted(data['time'],clock);inside=idx<len(data['time']);safe=np.minimum(idx,len(data['time'])-1)
    exact=inside&(data['time'][safe]==clock)&(data['segment'][safe]>=0)
    prices=(data['book'][safe,0]+data['book'][safe,2])/2
    values=[float(p) if ok and np.isfinite(p) else None for p,ok in zip(prices,exact)]
    if values[60] is None or not np.isclose(values[60],p0,rtol=1e-12):raise ValueError('Issued origin price mismatch')
    target=np.r_[origin,origin+np.array(HORIZONS)*1000].astype(np.int64)
    forecasts={}
    for method in METHODS:
        r=pred[method][row];q=np.array(widths[method])
        if r.shape!=(6,) or q.shape!=(6,) or not np.isfinite(r).all() or not np.isfinite(q).all() or (q<0).any():raise ValueError('Invalid forecast')
        forecasts[method]=dict(price=np.r_[p0,p0*np.exp(r)].tolist(),lower=np.r_[p0,p0*np.exp(r-q)].tolist(),upper=np.r_[p0,p0*np.exp(r+q)].tolist())
    endpoints=[values[int((t-clock[0])//1000)] for t in target]
    for j,p in enumerate(endpoints[1:]):
        a=pred['actual_returns'][row,j]
        if np.isfinite(a) and (p is None or not np.isclose(p0*np.exp(a),p,rtol=1e-12)):raise ValueError('Facts differ from scored outcomes')
    shown=[v for v in values if v is not None]+[v for f in forecasts.values() for v in f['price']]
    lo,hi=min(shown),max(shown);pad=max((hi-lo)*.08,p0*.000002)
    return dict(symbol=symbol,origin_ms=int(origin),origin_price=p0,clock_ms=clock.tolist(),price=values,target_ms=target.tolist(),
        actual_endpoints=endpoints,forecasts=forecasts,price_range=[lo-pad,hi+pad],scored=bool(pred['scored'][row]))


def static_plot(view,method,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as dates
    import pandas as pd
    x=pd.to_datetime(view['clock_ms'],unit='ms',utc=True);t=pd.to_datetime(view['target_ms'],unit='ms',utc=True)
    y=np.array([np.nan if v is None else v for v in view['price']]);f=view['forecasts'][method]
    fig,ax=plt.subplots(figsize=(12,5))
    ax.plot(x[:61],y[:61],color='#2875b8',label='Observed history')
    ax.plot(x[60:],y[60:],color='#e77d24',label='Actual AFTER issuance (not model input)')
    ax.plot(t,f['price'],'o--',color='#8054c3',lw=2,label='Frozen predictions: +5/+10/+15/+20/+25/+30 seconds')
    ax.axvline(t[0],color='black',ls=':',label='One issuance origin');ax.set_ylim(*view['price_range'])
    ax.set_title(f"{view['symbol']} | {method} | {t[0]:%Y-%m-%d %H:%M:%S} UTC\nSix model outputs from one origin; segments connect predicted endpoints")
    ax.set_ylabel('Binance perpetual mid-price, USDT');ax.set_xlabel('UTC');ax.grid(alpha=.2)
    ax.xaxis.set_major_formatter(dates.DateFormatter('%H:%M:%S',tz=t.tz));ax.legend(fontsize=8);fig.tight_layout();fig.savefig(path,dpi=140);plt.close(fig)


def metric_rows(result):
    rows=[]
    for method,groups in result['metrics'].items():
        for symbol,scores in [('pooled',groups['pooled'])]+list(groups['assets'].items()):
            for v in scores:
                if v['horizon_sec'] not in REGISTERED:continue
                rows.append(dict(model=method,asset=symbol,**v,direction_accuracy=v['direction_correct']/v['direction_n'] if v['direction_n'] else None,
                    majority_accuracy=v['observed_majority_correct']/v['direction_n'] if v['direction_n'] else None,coverage=v['interval_covered']/v['interval_n']))
    return rows


def metric_plot(result,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    controls=result['metrics']['Zero']['pooled'];names=list(METHODS)+['Microprice'];x=np.arange(3)
    fig,axes=plt.subplots(1,3,figsize=(16,5.5));colors=['#8054c3','#2875b8','#3a9763','#e77d24']
    for i,(name,color) in enumerate(zip(names,colors)):
        rows=[v for v in result['metrics'][name]['pooled'] if v['horizon_sec'] in REGISTERED]
        zero=[v for v in controls if v['horizon_sec'] in REGISTERED];offset=(i-1.5)*.2
        for ax,key in zip(axes[:2],('mae_bp','rmse_bp')):
            gain=[100*(1-v[key]/z[key]) if z[key] else np.nan for v,z in zip(rows,zero)]
            ax.bar(x+offset,gain,width=.19,label=name,color=color)
        axes[2].bar(x+offset,[100*v['direction_correct']/v['direction_n'] if v['direction_n'] else np.nan for v in rows],width=.19,label=name,color=color)
    for ax,title in zip(axes,('MAE improvement vs zero return, %','RMSE improvement vs zero return, %','Direction accuracy on nonzero moves, %')):
        ax.set_xticks(x,[f'+{h} sec' for h in REGISTERED]);ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    for ax in axes[:2]:ax.axhline(0,color='black',lw=1);ax.set_ylabel('Positive = smaller forecast error')
    rows=[v for v in controls if v['horizon_sec'] in REGISTERED]
    majority=[100*v['observed_majority_correct']/v['direction_n'] if v['direction_n'] else np.nan for v in rows]
    axes[2].plot(x,majority,'k--',label='Observed majority direction')
    values=majority+[100*v['direction_correct']/v['direction_n'] for name in names for v in result['metrics'][name]['pooled'] if v['horizon_sec'] in REGISTERED and v['direction_n']]
    finite=[v for v in values if np.isfinite(v)]
    axes[2].set_ylim(max(0,min(finite)-5) if finite else 0,min(100,max(finite)+5) if finite else 100);axes[2].legend(fontsize=8,loc='upper right')
    counts=', '.join(f"{v['horizon_sec']}s: {v['direction_n']:,}" for v in rows)
    fig.suptitle(f"BTC / ETH / SOL | same TEST cohort N={rows[0]['n']:,} | nonzero movement counts: {counts}",fontsize=11)
    fig.text(.02,.02,'Error gain = 100*(control error - model error)/control error. Exact errors, correct/N and class counts are in the report table.',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,.93]);fig.savefig(path,dpi=140);plt.close(fig)


def html_document(payload):
    from plotly.offline import get_plotlyjs
    data=json.dumps(payload,ensure_ascii=False,allow_nan=False).replace('</','<\\/')
    page='''<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>CatBoost order flow: 5/10/30 секунд</title>
<style>body{font:15px system-ui;max-width:1450px;margin:24px auto;padding:0 20px}header{position:sticky;top:0;background:white;z-index:10;padding:12px}select{padding:8px}table{border-collapse:collapse;display:block;overflow:auto}td,th{border:1px solid #ddd;padding:6px}.hint{color:#555}</style></head><body>
<h1>CatBoost на order flow: прогноз цены на 5, 10 и 30 секунд</h1>
<p>Фиолетовые точки — шесть прямых прогнозов цены из одной отсечки. Синий — доступная история; оранжевый — последующий факт.
Соединяющие отрезки показывают прогнозную кривую между выходами +5/+10/+15/+20/+25/+30 секунд; промежуточные тики модель не предсказывает.</p>
<p class="hint">Поток изменений первых пяти уровней стакана из ~100 мс состояний, без ленты сделок. Binance USD-M perpetual mid-price, март–апрель 2026.
Весь доступный архив: 90 исходных файлов. Все методы имеют одинаковые отсечки и проверяются на одних наблюдениях.
TEST уже раскрыт в предыдущих экспериментах: это ретроспективная проверка новой гипотезы, параметры фиксированы до новых результатов.</p>
<header>Монета <select id="asset"></select> Отсечка UTC <select id="origin"></select>
<label><input id="interval" type="checkbox"> 90% интервалы</label> <label><input id="focus" type="checkbox"> Увеличить будущее</label></header>
<div id="chart"></div><h2>Прогноз и последующий факт</h2><table id="facts"></table>
<h2>Сравнение качества на всей общей TEST-выборке</h2><img src="metrics_comparison.png" alt="MAE, RMSE и направление: сравнение методов на трёх горизонтах" style="width:100%">
<p>График объединяет три монеты; выбор монеты меняет таблицу ниже. Положительный прирост MAE/RMSE означает меньшую ошибку относительно нулевого изменения цены.</p>
<h2>Метрики на общей проверочной выборке</h2><p>MAE/RMSE — ошибка изменения цены в базисных пунктах. Accuracy — направление среди ненулевых фактических изменений.
Majority — доля преобладающего направления на этом TEST (описательный контроль). Нулевые прогнозы — отдельная категория; таблица отражает их количество.
Zero используется только для контроля ошибок. Microprice — текущий дисбаланс лучшей очереди. Дополнительно приведён счёт correct/N по трём знакам ↑/=/↓ на всех случаях.
Численный ноль — |log return| ≤ 1e−12, намного меньше ценового тика. Направление и доходность после спреда, комиссий и задержек различаются.</p><table id="metrics"></table>
<h2>Проверка устойчивости прироста OFI</h2><p>Парная разность MAE по полным UTC дням, bootstrap блоками по 3 дня, семейная поправка на 9 сравнений.
Менее 30 полных дней TEST не даёт достаточной опоры для устойчивого превосходства.</p><table id="paired"></table>
<h2>Данные и ограничения</h2><pre id="coverage"></pre>
<p>Интервалы построены по отдельной calibration-выборке; покрытие всей траектории не гарантировано.
Доступность определяется exchange E, независимых receive timestamps нет. Пропуски не дорисовываются.
Full Truth Harness: FAIL TH-11 — отсутствует корректный хеш источника канонического portfolio replay.
Эксперимент не меняет торговое поведение бота и не доказывает прибыльность.</p>
<p><a href="forecast_paths.json">Данные графиков</a> · <a href="metrics.csv">Метрики CSV</a> · <a href="result.json">Результат и происхождение</a></p>
<script>LIBRARY</script><script>const payload=PAYLOAD;
const asset=document.getElementById('asset'),origin=document.getElementById('origin'),iso=t=>new Date(t).toISOString();
payload.symbols.forEach(s=>asset.add(new Option(s,s)));payload.origins_ms.forEach(t=>origin.add(new Option(iso(t),String(t))));
const fmt=(v,n=4)=>v===null||v===undefined?'—':Number(v).toFixed(n);
const pct=v=>v===null||v===undefined?'—':fmt(100*v,2)+'%';
function table(id,headers,rows){document.getElementById(id).innerHTML='<tr>'+headers.map(h=>'<th>'+h+'</th>').join('')+'</tr>'+rows.map(r=>'<tr>'+r.map(v=>'<td>'+v+'</td>').join('')+'</tr>').join('');}
function draw(){const v=payload.views[asset.value][origin.value],tx=v.target_ms.map(iso),x=v.clock_ms.map(iso),focus=document.getElementById('focus').checked,interval=document.getElementById('interval').checked;
 const traces=[],shapes=[],annotations=[],shown=v.price.filter((p,i)=>p!==null&&(!focus||v.clock_ms[i]>=v.origin_ms));
 payload.methods.forEach(m=>{const f=v.forecasts[m];shown.push(...f.price);if(interval)shown.push(...f.lower,...f.upper);});
 const lo=Math.min(...shown),hi=Math.max(...shown),pad=Math.max((hi-lo)*.08,v.origin_price*.000002),range=[lo-pad,hi+pad];
 const layout={height:1250,margin:{l:90,r:30,t:55,b:55},hovermode:'x unified',showlegend:true,shapes,annotations};
 payload.methods.forEach((m,i)=>{const f=v.forecasts[m],a=i?'x'+(i+1):'x',b=i?'y'+(i+1):'y',axis={xaxis:a,yaxis:b};
  const add=(y,name,color,dash)=>traces.push({...axis,x,y,name,line:{color,dash},showlegend:i===0,connectgaps:false});
  add(v.price.map((p,k)=>v.clock_ms[k]<=v.origin_ms?p:null),'История','#2875b8','solid');
  add(v.price.map((p,k)=>v.clock_ms[k]>=v.origin_ms?p:null),'Факт после выдачи','#e77d24','solid');
  if(interval){traces.push({...axis,x:tx,y:f.lower,line:{width:0},showlegend:false,hoverinfo:'skip'});traces.push({...axis,x:tx,y:f.upper,line:{width:0},fill:'tonexty',fillcolor:'rgba(128,84,195,.13)',name:'90% interval',showlegend:i===0});}
  traces.push({...axis,x:tx,y:f.price,mode:'lines+markers',name:'Замороженный прогноз',line:{color:'#8054c3',dash:'dash',width:2},showlegend:i===0});
  const bottom=(2-i)/3+.025,top=(3-i)/3-.04;
  layout['xaxis'+(i?i+1:'')]={anchor:b,range:[iso(focus?v.origin_ms-5000:v.clock_ms[0]),iso(v.clock_ms.at(-1))],title:i===2?'UTC':''};
  layout['yaxis'+(i?i+1:'')]={anchor:a,domain:[bottom,top],range,title:'USDT'};
  shapes.push({type:'line',xref:a,yref:b,x0:iso(v.origin_ms),x1:iso(v.origin_ms),y0:range[0],y1:range[1],line:{dash:'dot',color:'black'}});
  annotations.push({xref:'paper',yref:'paper',x:.5,y:top+.02,text:m+' | '+v.symbol+' | '+iso(v.origin_ms),showarrow:false});
 });Plotly.react('chart',traces,layout,{responsive:true});
 table('facts',['+ секунд','Факт',...payload.methods],payload.horizons.map((h,j)=>[h,fmt(v.actual_endpoints[j+1]),...payload.methods.map(m=>fmt(v.forecasts[m].price[j+1]))]));
 const rows=payload.metrics.filter(r=>r.asset===asset.value||r.asset==='pooled');
 table('metrics',['Модель','Актив','+сек','N','MAE bp','RMSE bp','MAE USDT','Направление correct/N','Accuracy','Majority','↑/=/↓ correct/N','Факт ↓/0/↑','Прогноз ↓/0/↑','90% covered/N'],rows.map(r=>[r.model,r.asset,r.horizon_sec,r.n,fmt(r.mae_bp),fmt(r.rmse_bp),fmt(r.mae_USDT),r.direction_correct+'/'+r.direction_n,pct(r.direction_accuracy),pct(r.majority_accuracy),r.sign_correct+'/'+r.sign_n,r.actual_sign_counts.join('/'),r.predicted_sign_counts.join('/'),r.interval_covered+'/'+r.interval_n]));
}
table('paired',['Сравнение','Полных дней','Прирост MAE bp','Семейный 95% интервал','Устойчивое превосходство'],Object.entries(payload.paired).map(([k,v])=>[k,v.n_days,fmt(v.mean_mae_gain_bp),v.familywise95_bp?v.familywise95_bp.map(x=>fmt(x)).join(' … '):'—',v.robust_claim_allowed?'Да':'Нет']));
document.getElementById('coverage').textContent=JSON.stringify(payload.coverage,null,2);
asset.addEventListener('change',draw);origin.addEventListener('change',draw);document.getElementById('interval').addEventListener('change',draw);document.getElementById('focus').addEventListener('change',draw);draw();
</script></body></html>'''
    return page.replace('LIBRARY',get_plotlyjs()).replace('PAYLOAD',data)


def render(books,report):
    result=json.loads((report/'result.json').read_text(encoding='utf-8'));v=json.loads((report/'verification.json').read_text(encoding='utf-8'))
    if v['status']!='PASS' or sha(report/'result.json')!=v['result_sha256'] or sha(report/'predictions.npz')!=v['predictions_sha256']:raise ValueError('Frozen evidence mismatch')
    for name,checksum in v['native_sha256'].items():
        if sha(report/name)!=checksum:raise ValueError('Native model drift')
    if sha(books/'coverage.json')!=result['registration']['coverage_sha256']:raise ValueError('Source coverage drift')
    with np.load(report/'predictions.npz',allow_pickle=False) as f:pred={k:f[k] for k in f.files}
    origins=select_origins(pred,result['registration']['cuts']);views={};plots=report/'plots';plots.mkdir(exist_ok=True)
    for symbol in SYMBOLS:
        data=load_asset(books,symbol,result['coverage']);views[symbol]={str(t):build_view(data,pred,result['calibration_widths'],symbol,t) for t in origins}
        for method in METHODS:static_plot(views[symbol][str(origins[len(origins)//2])],method,plots/f'{symbol}_{method}.png')
        del data
    rows=metric_rows(result);fields=list(dict.fromkeys(k for row in rows for k in row))
    metric_plot(result,report/'metrics_comparison.png')
    with (report/'metrics.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    coverage={s:dict(**result['asset_coverage'][s],audit=result['coverage']['assets'][s]['audit']) for s in SYMBOLS}
    payload=dict(methods=list(METHODS),symbols=list(SYMBOLS),horizons=list(HORIZONS),origins_ms=origins,views=views,metrics=rows,paired=result['paired'],coverage=coverage,
        evidence=dict(result_sha256=v['result_sha256'],predictions_sha256=v['predictions_sha256'],renderer_sha256=sha(__file__),selection='fixed 12-hour clocks and past eligibility only'))
    (report/'forecast_paths.json').write_text(json.dumps(payload,indent=2,allow_nan=False),encoding='utf-8')
    (report/'forecast_paths.html').write_text(html_document(payload),encoding='utf-8');print('Rendered second-scale paths',len(origins),report,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--books',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();render(a.books,a.report)
