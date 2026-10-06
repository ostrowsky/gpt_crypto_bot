"""Execution evidence tables and diagnostic figures; no profitability claims."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from minute_direction_data import sha


def rows(result):
    out=[]
    for h,group in result['scores'].items():
        for method,r in group['methods'].items():
            out.append(dict(horizon_seconds=int(h),method=method,issued=r['issued'],deferred=r['deferred'],
                action_known=r['action_known'],action_unknown=r['action_unknown'],matched=r['matched'],
                matched_deferred=r['matched_deferred'],mean_gain_bp=r['mean_gain_bp'],selected_mean_gain_bp=r['selected_mean_gain_bp'],
                harmed=r['harmed'],benefited=r['benefited'],mean_shortfall_bp=r['mean_shortfall_bp'],
                p95_shortfall_bp=r['p95_shortfall_bp'],p99_shortfall_bp=r['p99_shortfall_bp']))
    return out


def render(output,spot):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result=json.loads((output/'result.json').read_bytes());audit=json.loads((output/'independent_verification.json').read_bytes())
    if audit['status']!='PASS' or audit['result_sha256']!=sha(output/'result.json'):raise ValueError('unverified report')
    flat=rows(result)
    with (output/'comparison.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    fig,axes=plt.subplots(3,1,figsize=(12,11),layout='constrained')
    colors={'OFI':'#c0392b','Book':'#2471a3','Microprice':'#239b56','AlwaysWait':'#777777'}
    hs=[5,10,30]
    for method,color in colors.items():
        group=[result['scores'][str(h)]['methods'][method] for h in hs]
        axes[0].plot(hs,[r['mean_gain_bp'] for r in group],'o-',color=color,label=method)
        axes[1].plot(hs,[r['p99_shortfall_bp'] for r in group],'o-',color=color,label=method)
        axes[2].plot(hs,[100*r['deferred']/r['issued'] for r in group],'o-',color=color,label=method)
    baseline=[result['scores'][str(h)]['methods']['Immediate']['p99_shortfall_bp'] for h in hs]
    axes[1].plot(hs,baseline,'k--',label='Immediate')
    axes[0].axhline(0,color='black',lw=1)
    for ax in axes:ax.grid(alpha=.2);ax.set_xticks(hs);ax.legend(ncol=5,fontsize=9)
    axes[0].set_ylabel('Mean gain vs immediate (bp)');axes[1].set_ylabel('p99 net shortfall (bp)');axes[2].set_ylabel('Deferred / issued (%)')
    axes[2].set_yscale('symlog',linthresh=.001)
    axes[2].set_yticks([0,.001,.01,.1,1,10,100],['0','0.001','0.01','0.1','1','10','100'])
    axes[2].set_xlabel('Maximum wait (seconds)')
    fig.suptitle('Order-flow execution diagnostic | 1000 USDT fixed-size tasks\nPerpetual event-time books; not spot fills or portfolio alpha')
    fig.savefig(output/'comparison.png',dpi=150);plt.close(fig)
    views=json.loads((output/'views.json').read_bytes());panels=views[:4]
    if panels:
        fig,axes=plt.subplots(len(panels),1,figsize=(12,3*len(panels)),squeeze=False,layout='constrained')
        for ax,v in zip(axes.ravel(),panels):
            t=np.array(v['clock']);origin=v['time'];prices=np.array([np.nan if p is None else p for p in v['mid']]);rel=(t-origin)/1000
            ax.plot(rel[rel<=0],prices[rel<=0],'k-',label='Observed before decision')
            ax.plot(rel[rel>=0],prices[rel>=0],color='#555',label='Later mid (not seen by model)')
            i=int(np.searchsorted(t,origin));p0=prices[i]
            ax.plot([0,5,10,30],[p0]+v['forecast_prices'],'o--',color='#d35400',label='Frozen OFI mid forecast')
            for j,h in enumerate(hs):
                if v['net_prices'][j] is not None:ax.scatter(v['delays'][j]+1,v['net_prices'][j],marker='x',s=60,label=f'h{h}: net arrival VWAP')
            ax.axvline(0,color='grey',ls=':');ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=3)
            utc=datetime.fromtimestamp(origin/1000,timezone.utc).isoformat()
            ax.set_title(f"{v['symbol']} | replay {v['kind']} | {utc} | transferred to perpetual quotes")
            ax.set_xlabel('Seconds from decision');ax.set_ylabel('USDT')
        fig.savefig(output/'order_examples.png',dpi=150);plt.close(fig)
    lines=['# Order flow для исполнения: NOT APPROVED','',
        'Ретроспективный диагностический тест. Цена со стаканом и комиссией; это не фактические fills и не доходность портфеля.',
        'Архив perpetual, бот spot. 1 секунда условной задержки прихода, размер 1000 USDT, комиссия 7.5 bp, без отдельного фиксированного slippage.',
        'Правило: ждать при signed forecast < −1 bp. Горизонт 5 секунд основной; 10/30 секунд вторичные. Модели заморожены до этого анализа, TEST уже был просмотрен.',
        '', '| h | Метод | Выдано | Отложено | Известны/неизвестны выбранные исходы | Сопоставимы | Выигрыш bp | p99 shortfall bp |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    fmt=lambda x:'unknown' if x is None else f'{x:.6f}'
    for r in flat:lines.append(f"| {r['horizon_seconds']} | {r['method']} | {r['issued']} | {r['deferred']} | {r['action_known']}/{r['action_unknown']} | {r['matched']} | {fmt(r['mean_gain_bp'])} | {fmt(r['p99_shortfall_bp'])} |")
    lines+=['','## Проверка устойчивости','',
        'Средние считаются только на парных доступных котировках, поэтому пропуски могут смещать оценку. Неизвестный исход не считается заполненной или пропущенной заявкой. BUY и SELL имеют равный диагностический вес; это не частоты действий бота.',
        'CI: 5000 bootstrap, блоки по 3 календарных дня, поправка на 9 сравнений. Внутренние дни архива содержат пробелы; 30 полных дней отсутствуют.']
    for h,g in result['scores'].items():
        for control,r in g['paired'].items():lines.append(f"- {h}s OFI vs {control}: {r['n_days']} дней; CI bp {r['familywise95_bp']}.")
    o=result['actual_orders'];lines+=['','## Максимальный период и исходные решения','',
        f"Проанализированы все {o['trades']} сделок, {o['issued']} заявочных действий из 186-дневного контрольного replay. Статусы: `{json.dumps(o['states'])}`.",
        'Перенос времени решений spot на perpetual является проверкой покрытия, а не доказательством улучшения реального исполнения. Hard и partial SELL не задерживаются. Данных для проверки основной метрики портфеля недостаточно; live-исполнение не меняется.']
    for h,r in o['scores'].items():lines.append(f"- {h}s: joined {r['joined']}/{r['issued']}, отложено {r['deferred']}, известны {r['known']}/{r['joined']}; gain bp {r['gains_bp']}.")
    pilot=json.loads((spot/'result.json').read_bytes());lines+=['','## Свежий spot-сбор','',
        f"120 секунд, `{pilot['status']}`; счётчики `{json.dumps(pilot['counts'])}`. Это пилот получения market data с local wall/monotonic receive timestamps, не тест модели и не fills.",
        'Интервалы между trade messages >500 ms могут означать отсутствие сделок, а не потерю данных. Depth exchange timestamps не выдумываются. Исторические receive timestamps восстановить таким сбором невозможно.',
        '', '## Итог','',
        'Тест завершён, производство не одобрено. Сравнение показывает диагностическое качество выбора момента, но не доказанный прирост назначения бота. Следующий gate — полный watchlist spot, связка decision/submit/ack/fill и >=30 свежих дней при неизменной политике.',
        f"Независимая проверка: PASS; native rows {audit['native_prediction_rows']}, scalar checks {audit['scalar_book_walk_checks']}, metric groups {audit['metric_groups']}. Prepared books producer-bound; scalar аудит выборочный.",
        'Первый исторический запуск остановлен до результатов: исправлена защита partial SELL; порог и модели не менялись. Первый spot-доступ заблокирован sandbox; повторный публичный сбор завершён.',
        '', '![Comparison](comparison.png)', '', '![Transferred orders](order_examples.png)']
    with (output/'report.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    with (output/'report_receipt.json').open('x',encoding='utf-8') as f:json.dump({p.name:sha(p) for p in output.iterdir() if p.name in ('report.md','comparison.csv','comparison.png','order_examples.png')},f,indent=2)
    print('RENDERED',output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--spot',type=Path,required=True)
    a=p.parse_args();render(a.output,a.spot)
