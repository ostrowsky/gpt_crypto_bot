"""Comparable full-period portfolio/cost plots and an honest research scorecard."""
import argparse
import csv
from datetime import datetime,timezone
import json
import math
from pathlib import Path

from historical_signal_evaluation import freeze,sha
from turnover_economics import ARMS


def exit_diagnostics(trades):
    retained=[];giveback=[]
    for t in trades:
        if (t.get('exit_price',0)>t.get('entry_price',0)>0
            and isinstance(t.get('exit_efficiency'),(int,float))
            and math.isfinite(t['exit_efficiency'])):
            retained.append(t['exit_efficiency'])
        if isinstance(t.get('giveback_pct'),(int,float)) and math.isfinite(t['giveback_pct']):
            giveback.append(t['giveback_pct'])
    return {'trades':len(trades),'winning_retention_n':len(retained),
        'winning_retention_mean':sum(retained)/len(retained) if retained else None,
        'giveback_n':len(giveback),'giveback_mean_pct':sum(giveback)/len(giveback) if giveback else None,
        'scope':'unweighted gross-price trade diagnostics, not after-cost portfolio return'}


def summary_rows(result):
    rows=[]
    for name in ARMS:
        a=result['accounts'][name];m=result['missions'][name]
        rows.append({'arm':name,'net_return_pct':a['net_return_pct'],'alpha_pp':a['alpha_pp'],
            'test_return_pct':a['test_return_pct'],'max_drawdown_pct':a['max_drawdown_pct'],
            'exposure_pct':a['average_gross_exposure_pct'],'trades':a['trades'],'replacements':a['replacements'],
            'early_captured':m['early_pair_count'],'captured':m['captured_pair_count'],'leader_pairs':m['label_pair_count'],
            'precision_n':m['objective_trade_count'],'precision_N':m['eligible_trade_count'],
            'numerical_gate':'CONTROL' if name=='control' else result['comparisons'][name]['numerical_gate'],
            'runtime_eligible':False})
    return rows


def render(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    r=json.loads((directory/'result.json').read_bytes());rows=summary_rows(r)
    exits={name:exit_diagnostics(json.loads((directory/f'trades_{name}.json').read_bytes())) for name in ARMS}
    freeze(directory/'exit_diagnostics.json',json.dumps(exits,indent=2,allow_nan=False).encode())
    with (directory/'comparison.csv').open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fig,axes=plt.subplots(2,1,figsize=(13,9),constrained_layout=True)
    for name in ARMS:
        curve=r['accounts'][name]['curve']
        times=[datetime.fromtimestamp(t/1000,timezone.utc) for t,v in curve]
        axes[0].plot(times,[v for t,v in curve],label=name)
    axes[0].set_yscale('log');axes[0].set_ylabel('Liquidation equity (USDT, log scale)')
    axes[0].set_title('Same cash account, full recovered period, fees/slippage included')
    axes[0].axvspan(datetime.fromtimestamp(r['split_boundaries_ms'][1]/1000,timezone.utc),
        datetime.fromtimestamp(r['end_ms']/1000,timezone.utc),color='gray',alpha=.12,label='Retrospective TEST')
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'));axes[0].legend();axes[0].grid(alpha=.25)
    groups=r['attribution']['control']['mode'];labels=sorted(groups)
    for offset,key in [(-.25,'raw_price_pnl'),(0,'fees'),(.25,'net_pnl')]:
        vals=[-groups[k][key]-groups[k]['slippage'] if key=='fees' else groups[k][key] for k in labels]
        axes[1].bar([i+offset for i in range(len(labels))],vals,width=.25,label='fees + slippage' if key=='fees' else key)
    axes[1].set_xticks(range(len(labels)),labels,rotation=15);axes[1].axhline(0,color='black',lw=.6)
    axes[1].set_ylabel('Additive quote PnL / costs (USDT)');axes[1].set_title('Control: after-cost allocated quantities, grouped by entry mode')
    axes[1].legend();axes[1].grid(axis='y',alpha=.25)
    fig.savefig(directory/'comparison.png',dpi=150);plt.close(fig)
    lines=['# Экономика входов и оборота: полный replay','',
        f"Период: {r['days']:.3f} дня; полные истории {r['population_complete']}/{r['population_requested']} символов.",
        'Все варианты пересчитаны с одними исходными кандидатами и прежними правилами выхода.',
        '','| Вариант | Net, % | Alpha к BTC, п.п. | TEST, % | DD, % | Сделки | Ранние лидеры | Gate |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for row in rows:
        lines.append(f"| {row['arm']} | {row['net_return_pct']:.2f} | {row['alpha_pp']:.2f} | {row['test_return_pct']:.2f} | {row['max_drawdown_pct']:.2f} | {row['trades']} | {row['early_captured']}/{row['leader_pairs']} | {row['numerical_gate']} |")
    lines+=['','Alpha, просадка и доходность рассчитаны по единому денежному счёту. TEST переоценивает непрерывный счёт от общей границы, включая перенесённые позиции.',
        'Снижение убытка или числа сделок не означает прибыльность либо сохранение цели бота.','',
        '## Проверка гипотез','']
    for name,c in r['comparisons'].items():
        failed=', '.join(k for k,v in c['checks'].items() if not v)
        lines.append(f"- {name}: **{c['numerical_gate']}**. Не пройдены: {failed or 'нет'}; исправленный интервал среднего дневного log-преимущества: {c['corrected_interval']} bp, {c['days']} полных TEST дней.")
    lines+=['','Все результаты ретроспективные. Runtime-переход не разрешён: отсутствует сертифицированная историческая популяция и паритет с live-исполнением.',
        '','## Денежная атрибуция контроля','',
        '| Режим входа | Сделки | Raw price PnL, USDT | Издержки, USDT | Net PnL, USDT |',
        '|---|---:|---:|---:|---:|']
    for key in labels:
        g=groups[key];lines.append(f"| {key} | {g['trades']} | {g['raw_price_pnl']:.2f} | {g['fees']+g['slippage']:.2f} | {g['net_pnl']:.2f} |")
    lines+=['','Raw price PnL использует реально распределённые в симуляции after-cost количества. Это аддитивная атрибуция; она отличается от отдельного бескомиссионного счёта с иным реинвестированием.',
        '','## Диагностика выходов','',
        '| Вариант | Среднее удержание MFE на gross-прибыльных сделках | Валидные случаи | Средний giveback, % | Валидные случаи |',
        '|---|---:|---:|---:|---:|']
    for name,e in exits.items():
        retained='нет меток' if e['winning_retention_mean'] is None else f"{e['winning_retention_mean']:.4f}"
        giveback='нет меток' if e['giveback_mean_pct'] is None else f"{e['giveback_mean_pct']:.4f}"
        lines.append(f"| {name} | {retained} | {e['winning_retention_n']}/{e['trades']} | {giveback} | {e['giveback_n']}/{e['trades']} |")
    lines+=['','Это невзвешенные метрики движения цены до издержек; они не заменяют денежный результат.',
        '',f"![Сравнение]({(directory.resolve()/'comparison.png').as_posix()})",'',
        'Манифесты, цены, признаки и кандидаты проверены по хешам. Полная кривая каждого счёта сверена с каноническим evaluator. Цены исполнения моделируются по закрытым свечам.']
    freeze(directory/'report.md',('\n'.join(lines)+'\n').encode('utf-8'))
    freeze(directory/'report_receipt.json',json.dumps({'result_sha256':sha(directory/'result.json'),
        'renderer_sha256':sha(Path(__file__)),**{n:sha(directory/n) for n in ('comparison.csv','comparison.png','report.md','exit_diagnostics.json')}}).encode())
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();print(json.dumps(render(a.run),indent=2))
