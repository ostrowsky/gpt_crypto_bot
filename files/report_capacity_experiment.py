"""Read-only experiment scorecard and transaction-cost hurdle diagnostics."""
from collections import Counter
import argparse
import json
import math
from pathlib import Path
from statistics import median

from historical_signal_evaluation import freeze, sha


def trade_diagnostics(trades, fee_bps=7.5, slippage_bps=5.):
    known=[];simple=[];invalid=0;partial=0
    hurdle=2*(fee_bps+slippage_bps)/100
    for t in trades:
        try:
            ep,xp=float(t['entry_price']),float(t['exit_price'])
            begin,end=int(t['entry_ts']),int(t['exit_ts'])
            if not all(math.isfinite(v) and v>0 for v in (ep,xp)) or end<begin:
                raise ValueError('unclosed/invalid trade')
        except (KeyError,TypeError,ValueError):invalid+=1;continue
        known.append((end-begin)/60000)
        if t.get('partial_exit_taken'):partial+=1;continue
        simple.append((xp/ep-1)*100)
    return {'trades_total':len(trades),'closed_known':len(known),'unknown':invalid,
        'partial_exits_excluded_from_hurdle':partial,'simple_full_exits':len(simple),
        'median_hold_minutes':median(known) if known else None,
        'roundtrip_cost_hurdle_pct':hurdle,
        'gross_positive_count':sum(v>0 for v in simple),
        'gross_move_exceeds_cost_count':sum(v>hurdle for v in simple),
        'gross_move_exceeds_double_cost_count':sum(v>2*hurdle for v in simple),
        'scope':'unweighted simulated trade diagnostics; not portfolio alpha or recorded fills'}


def build(directory):
    result=json.loads((directory/'result.json').read_bytes())
    trades=json.loads((directory/'baseline_trades.json').read_bytes())
    verification=json.loads((directory/'independent_verification.json').read_bytes())
    report={'verdict':result['verdict'],'runtime_eligible':False,
        'selection':{k:result[k] for k in ('groups','replacements','test_days','keep_mean_return_pct',
            'model_mean_return_pct','always_replace_mean_return_pct','paired_mean_uplift_pp','paired_3day_95ci_pp')},
        'transaction_cost_diagnostics':trade_diagnostics(trades),'verification':verification,
        'input_hashes':{name:sha(directory/name) for name in ('result.json','baseline_trades.json','independent_verification.json')},
        'report_source_sha256':sha(Path(__file__))}
    n=result['groups'];ci=result['paired_3day_95ci_pp'];cost=report['transaction_cost_diagnostics']
    text=f'''# Capacity CatBoost: проверка для бота

Результат: **{result['verdict']}**. Production-переход не разрешён этим экспериментом.

Максимальный восстановленный период: 181 день после разогрева, 93/105 полных
символов. Проверка: {n:,} конфликтов за {result['test_days']} дня с наблюдениями.
Это ретроспективная проверка на ранее использованном архиве.

| Выбор в конфликте | Средний результат следующих 75 минут |
|---|---:|
| Удерживать позицию | {result['keep_mean_return_pct']:+.6f}% |
| CatBoost Ranker | {result['model_mean_return_pct']:+.6f}% |
| Всегда заменять | {result['always_replace_mean_return_pct']:+.6f}% |

Это результат выбора между двумя вариантами, **не доходность портфеля**.
В результате нового кандидата учтены дополнительные издержки замены 0,25 п.п.
CatBoost выбрал замену в {result['replacements']}/{n} конфликтах. Среднее
преимущество относительно удержания: {result['paired_mean_uplift_pp']:+.6f} п.п.,
95% интервал [{ci[0]:+.6f}, {ci[1]:+.6f}] п.п.

На {result['keep_mission']['known_groups']} конфликтах с доступными дневными
метками выбранных лидеров: {result['model_mission']['leader_count']} у модели
против {result['keep_mission']['leader_count']} у контроля. Это повторяющиеся
выборы в конфликтах, не число уникальных ранних BUY бота.

## Диагностика оборота и издержек

Из {cost['trades_total']} симулированных сделок {cost['closed_known']} имеют
известный закрытый результат; {cost['unknown']} исключены как неизвестные.
Медиана удержания — {cost['median_hold_minutes']} минут.
Среди {cost['simple_full_exits']} полных выходов без частичной фиксации
положительное движение было у {cost['gross_positive_count']}, движение выше
порога издержек {cost['roundtrip_cost_hurdle_pct']:.2f}% — у
{cost['gross_move_exceeds_cost_count']}, выше удвоенного порога — у
{cost['gross_move_exceeds_double_cost_count']}.
Частичные выходы ({cost['partial_exits_excluded_from_hurdle']}) исключены из
этой простой оценки. Она не учитывает капитал каждой сделки и не заменяет
полную кривую счёта или проверку реальных исполнений.

## Проверка

Независимый численный аудит: {verification['status']}, все
{verification['native_prediction_rows']} прогнозов вариантов воспроизведены
из native-модели; {verification['raw_origin_audits']} фиксированных моментов
проверены по исходным свечам. Отдельная проверка Truth Harness относится
к каноническому baseline, а не к доказательству пользы этого ranker.

Следующий исследовательский приоритет: проверить экономику входов/выходов
и оборота на полной политике. Установка нового стека сама по себе не повышает
целевую метрику. Для promotion нужны полные данные, paired portfolio replay
и новые forward-наблюдения.
'''
    freeze(directory/'scorecard.json',json.dumps(report,indent=2,allow_nan=False).encode())
    freeze(directory/'scorecard.md',text.encode('utf-8'))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();print(json.dumps(build(a.run)['transaction_cost_diagnostics'],indent=2))
