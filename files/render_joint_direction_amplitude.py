"""Comparable account curves and independently verified probability calibration."""
import argparse
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
from historical_signal_evaluation import freeze,sha

ARMS=('control','regressor','joint')


def summary(r):
    rows=[]
    for name in ARMS:
        a=r['accounts'][name];m=r['missions'][name];t=r['test_missions'][name]
        rows.append(dict(arm=name,net_return_pct=a['net_return_pct'],alpha_pp=a['alpha_pp'],
            test_return_pct=a['test_return_pct'],drawdown_pct=a['max_drawdown_pct'],trades=a['trades'],
            exposure_pct=a['average_gross_exposure_pct'],early=m['early_pair_count'],captured=m['captured_pair_count'],
            leader_pairs=m['label_pair_count'],test_early=t['early_pair_count'],test_captured=t['captured_pair_count'],
            test_leader_pairs=t['label_pair_count'],precision_n=m['objective_trade_count'],precision_N=m['eligible_trade_count'],
            cost=a['totals']['fees']+a['totals']['slippage'],runtime_eligible=False))
    return rows


def render(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    r=json.loads((directory/'result.json').read_bytes());v=json.loads((directory/'independent_verification.json').read_bytes())
    if v['status']!='PASS' or v['result_sha256']!=sha(directory/'result.json'):raise ValueError('verified publication required')
    rows=summary(r);cut=r['split_boundaries_ms'][1]
    with (directory/'comparison.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    fig,axes=plt.subplots(3,1,figsize=(12,11),constrained_layout=True)
    for name in ARMS:
        curve=r['accounts'][name]['curve'];base=dict(curve)[cut]
        axes[0].plot([datetime.fromtimestamp(t/1000,timezone.utc) for t,p in curve],[p for t,p in curve],label=name)
        test=[(t,p) for t,p in curve if t>=cut]
        axes[1].plot([datetime.fromtimestamp(t/1000,timezone.utc) for t,p in test],[100*(p/base-1) for t,p in test],label=name)
    axes[0].set_yscale('log');axes[0].set_ylabel('Liquidation equity, USDT (log)')
    axes[0].axvspan(datetime.fromtimestamp(cut/1000,timezone.utc),datetime.fromtimestamp(r['end_ms']/1000,timezone.utc),color='gray',alpha=.12)
    axes[0].set_title('Maximum recovered period; after costs; regressor is prior rejected reference')
    axes[1].set_ylabel('Return from TEST origin, %');axes[1].set_title('Retrospective prequential TEST; carried holdings and capital')
    for ax in axes[:2]:ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    axes[2].plot([0,1],[0,1],color='gray',ls='--',label='Ideal calibration')
    for key in ('raw_probability','calibrated_probability','climatology'):
        bins=[b for b in r['forecast_metrics']['test'][key]['bins'] if b['n']]
        axes[2].plot([b['mean_probability'] for b in bins],[b['observed_up_rate'] for b in bins],marker='o',label=key)
    axes[2].set_xlim(0,1);axes[2].set_ylim(0,1);axes[2].set_xlabel('Predicted probability of RAW-price growth at +75m')
    axes[2].set_ylabel('Observed growth fraction');axes[2].set_title('TEST reliability: fixed ten bins; known outcomes only; counts in report')
    for ax in axes:ax.grid(alpha=.25);ax.legend()
    fig.savefig(directory/'comparison.png',dpi=150);plt.close(fig)
    comparison=r['comparisons']['joint']
    lines=['# Совместный прогноз направления и амплитуды','',
        f"Период {r['days']:.3f} дня; полная история {r['population_complete']}/{r['population_requested']} символов.",
        f"Фиксированная гипотеза: **{comparison['numerical_gate']}**. Production не изменён.",'',
        '| Вариант | Full net, % | Alpha BTC, п.п. | TEST net, % | DD, % | Сделки | Ранние лидеры | TEST ранние |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['arm']} | {row['net_return_pct']:.3f} | {row['alpha_pp']:.3f} | {row['test_return_pct']:.3f} | {row['drawdown_pct']:.3f} | {row['trades']} | {row['early']}/{row['leader_pairs']} | {row['test_early']}/{row['test_leader_pairs']} |")
    lines+=['','Regressor — прежний отклонённый вариант с неизменными прогнозами и сделками. Контроль и regressor пересчитаны по тому же денежному счёту; новое обучение и полный replay выполнены только для joint.',
        f"95% интервал среднего дневного log-преимущества joint над control на {comparison['days']} полных TEST днях: {comparison['interval']} bp.",
        'Не пройдены: '+', '.join(k for k,b in comparison['checks'].items() if not b),'',
        '## Вероятность роста','',
        '| Cohort / модель | Реальный рост / N | Правильное направление / N | Brier | Logloss | ECE10 |',
        '|---|---:|---:|---:|---:|---:|']
    for cohort,metrics in r['forecast_metrics'].items():
        for key in ('raw_probability','calibrated_probability','climatology'):
            a=metrics[key];lines.append(f"| {cohort} / {key} | {a['up_n']}/{a['n']} | {a['direction_correct_n']}/{a['n']} | {a['brier']:.6f} | {a['logloss']:.6f} | {a['ece10']:.6f} |")
    lines+=['','Direction здесь — правильный ответ «рост / отсутствие роста» через 75 минут до издержек; нулевая доходность относится к отсутствию роста. Climatology использует только долю роста в прошлой обучающей выборке. Высокая доля правильных ответов без преимущества над ней не доказывает полезности модели.',
        '','## Ожидаемый результат после издержек','',
        '| Cohort | Известные исходы / выданные | MAE joint | MAE flat price | MAE train mean | Положительные прогнозы с известным исходом | Их фактический средний net, % |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,a in r['forecast_metrics'].items():
        mean=a['accepted_realized_mean_net_pct'];text='нет' if mean is None else f'{mean:.6f}'
        lines.append(f"| {name} | {a['known_n']}/{a['issued_n']} | {a['net_mae']:.6f} | {a['flat_price_net_mae']:.6f} | {a['train_mean_net_mae']:.6f} | {a['accepted_known_n']} | {text} |")
    lines+=['','Это all-candidate фиксированные 75m исходы, а не исполненные сделки с прежними SELL правилами.',
        '','## TEST reliability: числители и знаменатели','',
        '| Модель / bin | N | Средняя вероятность | Реальный рост / N |','|---|---:|---:|---:|']
    for key in ('raw_probability','calibrated_probability','climatology'):
        for b in r['forecast_metrics']['test'][key]['bins']:
            mean='нет наблюдений' if b['mean_probability'] is None else f"{b['mean_probability']:.6f}"
            lines.append(f"| {key} / {b['bin']} | {b['n']} | {mean} | {b['up_n']}/{b['n']} |")
    lines+=['','## Ограничения','']+['- '+s for s in r['limitations']]
    lines+=['',f"Независимая проверка: {v['raw_rows']} сырых строк, {v['native_models']} native моделей, {v['reconstructed_calibrations']} повторных Platt-калибровок; все portfolio marks сверены.",
        '',f"![Портфели и калибровка]({(directory.resolve()/'comparison.png').as_posix()})",'']
    freeze(directory/'report.md',('\n'.join(lines)).encode('utf-8'))
    freeze(directory/'report_receipt.json',json.dumps({'result_sha256':sha(directory/'result.json'),
        'renderer_sha256':sha(Path(__file__)),**{n:sha(directory/n) for n in ('comparison.csv','comparison.png','report.md')}}).encode())
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();print(json.dumps(render(a.run),indent=2))
