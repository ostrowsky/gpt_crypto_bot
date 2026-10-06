"""Same-period portfolio curves plus concrete-action harm diagnostics."""
import argparse
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
from historical_signal_evaluation import freeze,sha

ARMS=('control','exit_model')


def summary(r):
    rows=[]
    for name in ARMS:
        a=r['accounts'][name];m=r['missions'][name];t=r['test_missions'][name]
        rows.append(dict(arm=name,net_return_pct=a['net_return_pct'],alpha_pp=a['alpha_pp'],
            test_return_pct=a['test_return_pct'],drawdown_pct=a['max_drawdown_pct'],trades=a['trades'],
            exposure_pct=a['average_gross_exposure_pct'],early=m['early_pair_count'],captured=m['captured_pair_count'],
            leader_pairs=m['label_pair_count'],test_early=t['early_pair_count'],test_captured=t['captured_pair_count'],
            test_leader_pairs=t['label_pair_count'],precision_n=m['objective_trade_count'],precision_N=m['eligible_trade_count'],
            fees=a['totals']['fees'],slippage=a['totals']['slippage'],runtime_eligible=False))
    return rows


def render(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import numpy as np
    r=json.loads((directory/'result.json').read_bytes());v=json.loads((directory/'independent_verification.json').read_bytes())
    if v['status']!='PASS' or v['result_sha256']!=sha(directory/'result.json'):raise ValueError('verified publication required')
    rows=summary(r);cut=r['split_boundaries_ms'][1]
    with (directory/'comparison.csv').open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fig,axes=plt.subplots(3,1,figsize=(12,10),constrained_layout=True)
    for name in ARMS:
        curve=r['accounts'][name]['curve'];base=dict(curve)[cut]
        axes[0].plot([datetime.fromtimestamp(t/1000,timezone.utc) for t,p in curve],[p for t,p in curve],label=name)
        test=[(t,p) for t,p in curve if t>=cut]
        axes[1].plot([datetime.fromtimestamp(t/1000,timezone.utc) for t,p in test],[100*(p/base-1) for t,p in test],label=name)
    axes[0].set_yscale('log');axes[0].set_ylabel('Liquidation equity, USDT (log)')
    axes[0].axvspan(datetime.fromtimestamp(cut/1000,timezone.utc),datetime.fromtimestamp(r['end_ms']/1000,timezone.utc),color='gray',alpha=.12)
    axes[0].set_title('Maximum recovered period; costs included; BUY and cooldown rules unchanged')
    axes[1].set_ylabel('Return from TEST origin, %');axes[1].set_title('Retrospective prequential TEST; original capital and holdings carried')
    for ax in axes[:2]:ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'));ax.legend();ax.grid(alpha=.25)
    with np.load(directory/'dataset.npz',allow_pickle=False) as z:
        selected=(z['clock']>=cut)&(z['fold']>=0)&(z['prediction']>.05)&np.isfinite(z['y'])
        x=z['prediction'][selected];y=z['y'][selected]
    if len(x):axes[2].scatter(x,y,s=24,label=f'{len(x)} selected known control-exit states')
    axes[2].axhline(0,color='gray',ls='--');axes[2].set_xlabel('Predicted protected-hold advantage, pp')
    axes[2].set_ylabel('Realized isolated action advantage, pp');axes[2].set_title('Single-position diagnostic; capacity opportunity cost evaluated by full account')
    axes[2].grid(alpha=.25)
    if len(x):axes[2].legend()
    fig.savefig(directory/'comparison.png',dpi=150);plt.close(fig)
    c=r['comparisons']['exit_model'];lines=['# Модель денежной ценности отложенного SELL','',
        f"Период {r['days']:.3f} дня; полная история {r['population_complete']}/{r['population_requested']} символов.",
        f"Фиксированная гипотеза: **{c['numerical_gate']}**. Production не изменён.",'',
        '| Вариант | Full net, % | Alpha BTC, п.п. | TEST net, % | DD, % | Сделки | Ранние лидеры | TEST ранние |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['arm']} | {row['net_return_pct']:.4f} | {row['alpha_pp']:.4f} | {row['test_return_pct']:.4f} | {row['drawdown_pct']:.4f} | {row['trades']} | {row['early']}/{row['leader_pairs']} | {row['test_early']}/{row['test_leader_pairs']} |")
    lines+=['',f"95% интервал среднего дневного log-преимущества на {c['days']} полных TEST днях: {c['interval']} bp.",
        'Не пройдены: '+', '.join(k for k,b in c['checks'].items() if not b),'',
        f"В полном новом replay: {r['policy_counts']['actual_deferrals']} отсрочек на {r['policy_counts']['actual_soft_decisions']} фактических первых мягких SELL.",
        'Отсрочка действует максимум одну свечу; исходные hard exits продолжают работать. Deadline сохраняет исходную WEAK-категорию и соответствующий cooldown.',
        '','## Проверка конкретного действия на состояниях контроля','',
        '| Cohort | Известные / выданные | Отобранные известные | Ухудшения / отобранные | Средняя дельта, п.п. | Медиана | P10 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    def number(value):return 'нет' if value is None else f'{value:.6f}'
    for name,a in r['action_diagnostics'].items():
        lines.append(f"| {name} | {a['known_n']}/{a['issued_n']} | {a['selected_known_n']} | {a['selected_hurt_n']}/{a['selected_known_n']} | {number(a['selected_mean_advantage_pp'])} | {number(a['selected_median_advantage_pp'])} | {number(a['selected_p10_pp'])} |")
    lines+=['','Это разница денежных поступлений от одной продажи сейчас либо позже, нормированная на текущую цену. Она не равна приросту портфельной доходности. Состояния контрольных выходов и фактические решения изменённой политики имеют разные распределения.',
        '','## Удержание прибыли и giveback','',
        '| Вариант | Средняя winning MFE retention | Известные случаи | Средний giveback, % | Известные случаи |',
        '|---|---:|---:|---:|---:|']
    for name,a in r['exit_diagnostics'].items():
        lines.append(f"| {name} | {number(a['winning_retention_mean'])} | {a['winning_retention_n']}/{a['trades']} | {number(a['giveback_mean_pct'])} | {a['giveback_n']}/{a['trades']} |")
    lines+=['','Метрики MFE/giveback — невзвешенные gross-диагностики, денежный результат показан отдельно.',
        '','## Ограничения','']+['- '+s for s in r['limitations']]
    lines+=['',f"Независимая проверка: {v['dataset_soft_exits']} контрольных soft exit состояний, {v['native_control_predictions']} native прогнозов, {v['actual_soft_decisions']} фактических soft решений; все portfolio marks сверены.",
        '','Первая интеграция v1 сохранена как superseded до портфельных результатов: имя deadline ошибочно меняло cooldown-категорию. v2 исправляет этот контракт без изменения моделей или порога.',
        '',f"![Портфели и действие]({(directory.resolve()/'comparison.png').as_posix()})",'']
    freeze(directory/'report.md',('\n'.join(lines)).encode('utf-8'))
    freeze(directory/'report_receipt.json',json.dumps({'result_sha256':sha(directory/'result.json'),
        'renderer_sha256':sha(Path(__file__)),**{n:sha(directory/n) for n in ('comparison.csv','comparison.png','report.md')}}).encode())
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();print(json.dumps(render(a.run),indent=2))
