"""Full and TEST equity comparison with explicit capture guardrails."""
import argparse
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
from historical_signal_evaluation import freeze,sha


def summary(r):
    rows=[]
    for name in ('control','catboost'):
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
    r=json.loads((directory/'result.json').read_bytes());v=json.loads((directory/'independent_verification.json').read_bytes())
    if v['status']!='PASS' or v['result_sha256']!=sha(directory/'result.json'):raise ValueError('verified publication required')
    rows=summary(r);cut=r['split_boundaries_ms'][1]
    with (directory/'comparison.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    fig,axes=plt.subplots(2,1,figsize=(12,8),constrained_layout=True)
    for name in ('control','catboost'):
        curve=r['accounts'][name]['curve'];base=dict(curve)[cut]
        ts=[datetime.fromtimestamp(t/1000,timezone.utc) for t,p in curve]
        axes[0].plot(ts,[p for t,p in curve],label=name)
        test=[(t,p) for t,p in curve if t>=cut]
        axes[1].plot([datetime.fromtimestamp(t/1000,timezone.utc) for t,p in test],[100*(p/base-1) for t,p in test],label=name)
    axes[0].set_yscale('log');axes[0].set_ylabel('Liquidation equity, USDT (log)')
    axes[0].axvspan(datetime.fromtimestamp(cut/1000,timezone.utc),datetime.fromtimestamp(r['end_ms']/1000,timezone.utc),color='gray',alpha=.12)
    axes[0].set_title('Maximum recovered period; initial 30 days same policy; causal model refits')
    axes[1].set_ylabel('Return from common TEST origin, %');axes[1].set_title('Retrospective prequential TEST; carried holdings/capital; costs included')
    for ax in axes:ax.legend();ax.grid(alpha=.25);ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    fig.savefig(directory/'comparison.png',dpi=150);plt.close(fig)
    c=r['comparisons']['catboost'];lines=['# CatBoost: качество входов impulse_speed','',
        f"Период {r['days']:.3f} дня, полная история {r['population_complete']}/{r['population_requested']} символов.",
        f"Результат фиксированной гипотезы: **{c['numerical_gate']}**. Production не изменён.",'',
        '| Вариант | Full net, % | Alpha BTC, п.п. | TEST net, % | DD, % | Сделки | Ранние лидеры | TEST ранние |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for a in rows:
        lines.append(f"| {a['arm']} | {a['net_return_pct']:.3f} | {a['alpha_pp']:.3f} | {a['test_return_pct']:.3f} | {a['drawdown_pct']:.3f} | {a['trades']} | {a['early']}/{a['leader_pairs']} | {a['test_early']}/{a['test_leader_pairs']} |")
    lines+=['',f"95% интервал среднего дневного log-преимущества на {c['days']} полных TEST днях: {c['interval']} bp.",
        'Не пройдены: '+', '.join(k for k,b in c['checks'].items() if not b),
        '',f"Фильтр: {r['filter_audit']}",'',
        'Модель предсказывает фиксированную 75-минутную доходность после издержек. Реальные правила SELL остаются прежними: качество этой прокси не заменяет портфельный результат.',
        '','| Forecast cohort | Известные исходы / выданные | MAE, % | MAE нулевого прогноза, % | Принятые с известным исходом | Средний net proxy, % |',
        '|---|---:|---:|---:|---:|---:|']
    for name,f in r['forecast_metrics'].items():
        mean=f['accepted_mean_net_proxy_pct'];mean='нет' if mean is None else f'{mean:.5f}'
        lines.append(f"| {name} | {f['label_known_n']}/{f['issued_n']} | {f['mae_pct']:.5f} | {f['zero_mae_pct']:.5f} | {f['accepted_n']} | {mean} |")
    lines+=['','Каждый блок прогнозируется моделью, обученной на уже созревших прошлых исходах. Доступные исходы предыдущих TEST блоков могут попадать в последующее обучение: это последовательная OOS-проверка, не запечатанный holdout.',
        '','## Ограничения','']+['- '+s for s in r['limitations']]
    lines+=['',f"Независимая проверка: {v['raw_impulse_rows']} сырых строк, {v['native_predictions']} native прогнозов; каждый portfolio mark сверён с каноническим evaluator.",
        '',f"![Сравнение портфелей]({(directory.resolve()/'comparison.png').as_posix()})",'']
    freeze(directory/'report.md',('\n'.join(lines)).encode('utf-8'))
    freeze(directory/'report_receipt.json',json.dumps({'result_sha256':sha(directory/'result.json'),
        'renderer_sha256':sha(Path(__file__)),**{n:sha(directory/n) for n in ('comparison.csv','comparison.png','report.md')}}).encode())
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();print(json.dumps(render(a.run),indent=2))
