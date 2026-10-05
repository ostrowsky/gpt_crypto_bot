"""Render completed offline evidence; no model selection or production approval."""
import argparse,json,html,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
NAMES={'baseline':'Правила бота','volatility':'+ прогноз волатильности','direction':'+ прогноз доходности','combined':'+ оба прогноза','ewma_control':'Контроль: EWMA-размер'}

def validate(data):
    if data['status']!='COMPLETED_RETROSPECTIVE_DIAGNOSTIC' or data['runtime_eligible'] or data['achievement_claimed']:
        raise ValueError('Not completed offline evidence')
    if set(data['accounts'])!=set(NAMES) or data['benchmark']['status']!='complete':raise ValueError('Missing arms/benchmark')
    start,end=data['start_ms'],data['end_ms'];expected=np.arange(start,end+1,900000)
    for name,row in data['accounts'].items():
        curve=np.asarray(row['curve'])
        if len(curve)!=len(expected) or not np.array_equal(curve[:,0],expected) or not np.isfinite(curve).all() or (curve[:,1]<=0).any():
            raise ValueError('Incomplete equity curve')
        if row['max_positions']>10:raise ValueError('Invalid portfolio capacity')
        capture=data['captures'][name]
        if capture['captured_pair_count']>capture['label_pair_count']:raise ValueError('Invalid capture numerator')
    if any(f['last_training_label']>=f['start'] for f in data['folds']):raise ValueError('Leaking training boundary')

def comparison_rows(data):
    validate(data);btc=data['benchmark']['net_return_after_costs_pct'];rows=[]
    for name,row in data['accounts'].items():
        obj=data['captures'][name]
        n=obj['label_pair_count'];captured=obj['captured_pair_count'];trades=obj['eligible_trade_count']
        rows.append(dict(variant=NAMES[name],return_pct=row['net_return_pct'],btc_alpha_pp=row['net_return_pct']-btc,
            max_drawdown_pct=row['max_drawdown_pct'],exposure_pct=row['average_gross_exposure_pct'],trades=row['trades'],
            captured=f'{captured}/{n}',capture_pct=100*captured/n if n else None,
            early=f"{obj['early_pair_count']}/{n}",false_BUY=f"{trades-obj['objective_trade_count']}/{trades}",
            stress_return_pct=data['cost_stress'][name]['net_return_pct']))
    return rows

def load_evidence(folder):
    raw_path=folder/'result.json';raw=json.loads(raw_path.read_bytes())
    if not (folder/'analysis.json').exists():return raw
    derived=json.loads((folder/'analysis.json').read_bytes())
    if derived['source_result_sha256']!=hashlib.sha256(raw_path.read_bytes()).hexdigest():raise ValueError('Raw result drift')
    if derived['derivation_kind']!='starting-capital-statistics-repair':raise ValueError('Unknown statistics repair')
    if any(derived.get(k)!=v for k,v in raw.items() if k!='paired_intervals'):raise ValueError('Repair changed frozen experiment')
    for name in ('compare_price_volatility_bot.py','render_price_volatility_comparison.py'):
        if derived['derivation_source_hashes'][name]!=hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest():
            raise ValueError('Derivation source drift')
    from compare_price_volatility_bot import paired_intervals
    if derived['paired_intervals']!=paired_intervals(raw['accounts']):raise ValueError('Derived intervals do not match frozen curves')
    return derived

def render(folder):
    data=load_evidence(folder);rows=comparison_rows(data)
    table=pd.DataFrame(rows).rename(columns={
        'variant':'Вариант','return_pct':'Доходность, %','btc_alpha_pp':'К BTC, п.п.',
        'max_drawdown_pct':'Max DD, %','exposure_pct':'Средняя экспозиция, %',
        'trades':'Сделки','captured':'Лидеры роста','capture_pct':'Захват, %',
        'early':'Ранние лидеры','false_BUY':'BUY вне top-10',
        'stress_return_pct':'Двойные затраты: доходность, %'})
    utc=lambda v:pd.Timestamp(v,unit='ms',tz='UTC').isoformat()
    text=[f"# Прогноз доходности и волатильности: сравнение для криптобота\n",
          f"Период UTC: {utc(data['start_ms'])} — {utc(data['end_ms'])}; {data['days']:.3f} дня. "
          f"Полная история: {data['population_complete']}/{data['population_requested']} монет.\n",
          f"Комиссия {data['costs']['fee_bps']} bp и проскальзывание {data['costs']['slippage_bps']} bp на каждую сторону. "
          f"BTC buy-and-hold после затрат: {data['benchmark']['net_return_after_costs_pct']:.3f}%.\n",
          "Это текущие правила с отключёнными несвязанными историческими ML-весами; не полная реконструкция работающего бота. "
          "Варианты зафиксированы до просмотра результатов. Прогноз — следующий час, RV-прокси — четыре будущие 15m доходности.\n",
          '| Вариант | Доходность, % | К BTC, п.п. | Max DD, % | Средняя экспозиция, % | Сделки | Лидеры роста | Ранние лидеры | BUY вне дневного top-10 | Доходность при двойных затратах, % |',
          '|---|---:|---:|---:|---:|---:|---|---|---|---:|']
    for r in rows:text.append(f"| {r['variant']} | {r['return_pct']:.3f} | {r['btc_alpha_pp']:.3f} | {r['max_drawdown_pct']:.3f} | {r['exposure_pct']:.3f} | {r['trades']} | {r['captured']} | {r['early']} | {r['false_BUY']} | {r['stress_return_pct']:.3f} |")
    winner=max(rows,key=lambda r:r['return_pct']);risk=min(rows,key=lambda r:r['max_drawdown_pct'])
    text+=['','BUY вне top-10 — сделки в монетах, не вошедших в дневной top-10 к 22:00 Europe/Budapest; это не число убыточных сделок.',
            '',f"Численный лидер доходности: **{winner['variant']}**. Минимальная просадка: **{risk['variant']}**. "
            "Это разные критерии; меньшая экспозиция сама может уменьшать просадку.",
            '', '## Парное сравнение дневной доходности',
            'Интервалы — средняя разница дневной log-доходности против правил бота, basis points; 5000 bootstrap, поправка на три сравнения. '
            'Неполный последний день включён в итоговый портфель, но исключён из дневного bootstrap.']
    for name,r in data['paired_intervals'].items():
        text.append(f"- {NAMES[name]}: {r['n_days']} полных дней; блок 1 день {r['blocks']['1']['familywise95']}; блок 3 дня {r['blocks']['3']['familywise95']}.")
    metrics=data['forecast_metrics']
    control_path=folder/'forecast_control_diagnostics.json'
    control=json.loads(control_path.read_bytes()) if control_path.exists() else None
    if control and (control['n']!=metrics['n'] or control['always_up_correct']!=metrics['always_up_correct']):
        raise ValueError('Control metric denominator differs')
    text+=['','## Качество прогнозов',f"N={metrics['n']} непересекающихся часовых окон; правильное направление {metrics['direction_correct']}/{metrics['n']}, "
           f"контроль всегда вверх {metrics['always_up_correct']}/{metrics['n']}. MAE log-return {metrics['return_MAE']:.7f}, контроль последней цены {metrics['control_MAE']:.7f}.",
           f"QLIKE (меньше лучше): HAR {metrics['qlike']['har']:.6f}, прошлое RV {metrics['qlike']['past_rv']:.6f}, EWMA {metrics['qlike']['ewma']:.6f}. "
           f"Часовых RV ниже численного пола 1e-12: {metrics['variance_floor_count']}/{metrics['n']}.",
           '', '## Ограничения',*[f'- {x}' for x in data['limitations']],
           '- Ни численный лидер, ни положительный интервал не дают разрешения менять BUY/SELL. Нужны полная PIT/execution parity и новый forward/shadow период.',
           '- Полный Truth Harness: FAIL TH-11. Этот пересчёт не заменяет канонический портфельный артефакт.',
           '', '![Портфель и просадка](comparison.png)']
    if control:
        text.insert(text.index('## Ограничения'),
            f"Контроль всегда вниз: {control['always_down_correct']}/{control['n']}; фактически нулевая часовая доходность: {control['flat']}/{control['n']}. "
            f"Прогнозы дисперсии ниже пола 1e-12: {control['predicted_variance_floor_counts']}. "
            'У прошлого RV нулевые прогнозы делают QLIKE огромным; практический контроль здесь — EWMA. Окна разных монет коррелируют, N не означает столько независимых наблюдений.\n')
    if 'derivation_kind' in data:
        text+=['','Статистическая поправка: первый день bootstrap начинается с исходного капитала 1 и включает затраты входов в момент старта. '
               'Исходный result.json, прогнозы, сделки, кривые и их source hashes сохранены. В analysis.json записаны hash исходного отчёта и версии кода пересчёта.']
    (folder/'comparison.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(14,9),sharex=True)
    colors=['black','tab:blue','tab:orange','tab:green','tab:purple']
    for color,(name,row) in zip(colors,data['accounts'].items()):
        a=np.asarray(row['curve']);times=pd.to_datetime(a[:,0],unit='ms',utc=True);value=a[:,1]
        axes[0].plot(times,(value-1)*100,label=NAMES[name],color=color,lw=1.3)
        axes[1].plot(times,100*(value/np.maximum(1.0,np.maximum.accumulate(value))-1),label=NAMES[name],color=color,lw=1.2)
    axes[0].axhline(0,color='gray',lw=.5);axes[0].set_ylabel('Доходность после затрат, %');axes[0].legend(fontsize=9)
    axes[1].set_ylabel('Просадка, %');axes[1].set_xlabel('UTC')
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle(f"Ретроспективное сравнение · {data['population_complete']} монеты · {data['days']:.1f} дня · без допуска в production")
    fig.tight_layout();fig.savefig(folder/'comparison.png',dpi=140);plt.close(fig)
    # Standalone interactive curve explorer, no network dependencies.
    curves=[]
    for color,(name,row) in zip(['#111','#1976d2','#ef6c00','#2e7d32','#7b1fa2'],data['accounts'].items()):
        a=np.asarray(row['curve']);equity=a[:,1];values=(equity-1)*100;dd=(equity/np.maximum(1.0,np.maximum.accumulate(equity))-1)*100
        curves.append(dict(name=NAMES[name],color=color,x=a[:,0].tolist(),ret=values.tolist(),dd=dd.tolist()))
    heading=html.escape(f"Сравнение: {utc(data['start_ms'])} → {utc(data['end_ms'])}")
    document='''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Прогнозы для криптобота</title>
<style>body{font:15px system-ui;max-width:1300px;margin:30px auto;padding:0 20px;color:#17202a}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}th:first-child,td:first-child{text-align:left}canvas{width:100%;height:420px;border:1px solid #ddd;margin-top:15px}label{margin-right:15px}select{padding:6px}.note{background:#f2f5f8;padding:15px}</style>
<h1>Прогноз доходности и волатильности</h1><p>HEADING</p><p class="note">Ретроспективный walk-forward на фиксированной полной части архива. Историческая live/PIT parity не подтверждена. Full Truth Harness: FAIL TH-11. Торговые правила не изменены.</p>TABLE
<p>Метрика: <select id="metric"><option value="ret">Доходность после затрат, %</option><option value="dd">Просадка, %</option></select></p><div id="controls"></div><canvas id="chart" width="1200" height="420"></canvas><p id="cursor">Наведите мышь на график для сравнения в один момент.</p>
<p>Четыре основных варианта и EWMA-контроль размера. Прогноз доходности — Ridge, волатильности — HAR-style Ridge. Горизонт 1 час, 15m данные. Все интервалы и ограничения — в comparison.md.</p>
<script>const curves=CURVES,canvas=document.querySelector('#chart'),ctx=canvas.getContext('2d'),controls=document.querySelector('#controls');
curves.forEach((s,i)=>{let l=document.createElement('label');l.innerHTML='<input type="checkbox" checked data-i="'+i+'"> <span style="color:'+s.color+'">'+s.name+'</span>';controls.append(l)});
let state;function draw(){state=null;let key=document.querySelector('#metric').value,sel=[...controls.querySelectorAll('input')].filter(c=>c.checked).map(c=>curves[+c.dataset.i]);ctx.clearRect(0,0,1200,420);if(!sel.length)return;
let vals=sel.flatMap(s=>s[key]),lo=Math.min(0,...vals),hi=Math.max(0,...vals),span=hi-lo||1,x0=curves[0].x[0],x1=curves[0].x.at(-1);state={key,sel,lo,hi,span,x0,x1};ctx.font='12px system-ui';
for(let j=0;j<=5;j++){let y=30+j*65;ctx.strokeStyle='#e3e7eb';ctx.beginPath();ctx.moveTo(65,y);ctx.lineTo(1180,y);ctx.stroke();ctx.fillStyle='#555';ctx.fillText((hi-j*span/5).toFixed(1)+'%',8,y+4)}
sel.forEach(s=>{ctx.strokeStyle=s.color;ctx.lineWidth=1.5;ctx.beginPath();s.x.forEach((x,i)=>{let px=65+(x-x0)/(x1-x0)*1115,py=30+(hi-s[key][i])/span*325;i?ctx.lineTo(px,py):ctx.moveTo(px,py)});ctx.stroke()});
for(let j=0;j<=5;j++){let at=x0+(x1-x0)*j/5;ctx.fillStyle='#555';ctx.fillText(new Date(at).toISOString().slice(0,10),65+1115*j/5-30,390)}}
controls.onchange=draw;document.querySelector('#metric').onchange=draw;
canvas.onmousemove=e=>{if(!state)return;let x=(e.clientX-canvas.getBoundingClientRect().left)/canvas.getBoundingClientRect().width*1200;let f=Math.max(0,Math.min(1,(x-65)/1115)),i=Math.round(f*(curves[0].x.length-1));document.querySelector('#cursor').textContent=new Date(curves[0].x[i]).toISOString()+' · '+state.sel.map(s=>s.name+': '+s[state.key][i].toFixed(3)+'%').join(' · ')};draw();</script></html>'''
    # 90k samples exceed JavaScript spread argument limits; y-limits use reduce.
    document=document.replace('Math.min(0,...vals)','vals.reduce((a,b)=>Math.min(a,b),0)').replace('Math.max(0,...vals)','vals.reduce((a,b)=>Math.max(a,b),0)')
    document=document.replace('HEADING',heading).replace('TABLE',table.round(3).to_html(index=False)).replace('CURVES',json.dumps(curves,allow_nan=False))
    (folder/'comparison.html').write_text(document,encoding='utf-8')
    print(folder/'comparison.html')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);render(p.parse_args().folder)
