"""Authoritative calendar-corrected research summary; no trading approval."""
import argparse,json,hashlib
from pathlib import Path
from mission_calendar_evaluation import interval,evaluate
from audit_mission_learning import ENTRY_FIELDS

def load(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ratio(n,d):return f'{n}/{d} ({100*n/d:.2f}%)' if d else f'{n}/{d} (не определено)'

def publish(experiment):
    raw=load(experiment/'result.json');audit=load(experiment/'independent_verification.json')
    if audit['status']!='PASS' or audit['result_sha256']!=sha(experiment/'result.json'):raise ValueError('independent audit not bound to raw result')
    prefix=load(experiment/'prefix_verification.json')
    if prefix['status']!='PASS' or prefix['result_sha256']!=sha(experiment/'result.json'):raise ValueError('candidate prefix proof not bound')
    days=load(experiment/'daily_metrics.json');ci=interval(days['control'],days['early_ranker'])
    base,target=(raw['arms'][n]['test'] for n in ('control','early_ranker'));gate=evaluate(base,target,ci,False)
    paired={}
    for name in ('control','early_ranker'):
        episodes=load(experiment/f'episodes_{name}.json');trades=load(experiment/f'trades_{name}.json');by_entry={(r['sym'],r['entry_ts'],r['entry_price']):r for r in trades}
        paired[name]={(e['day'],e['symbol']):(e,by_entry[e['symbol'],e['entry_ts'],e['entry_price']]) for e in episodes if e['state']=='CONFIRMED' and not e['terminal_forced'] and e['day']>=min(r['day'] for r in days[name])}
    common=paired['control'].keys()&paired['early_ranker'].keys();strict=0
    for k in common:
        (_,a),(_,b)=paired['control'][k],paired['early_ranker'][k]
        strict+=all(a[f]==b[f] for f in ENTRY_FIELDS)
    companion=dict(state='UNVERIFIED',common_confirmed_pairs=len(common),matched_all16_entry_fields=strict,scope='timestamps/price alone insufficient; legacy paired_exit_comparison not approved')
    corrected=dict(contract='mission-calendar-evaluation-v1',raw_result_sha256=sha(experiment/'result.json'),audit_sha256=sha(experiment/'independent_verification.json'),prefix_sha256=sha(experiment/'prefix_verification.json'),calendar_interval=ci,gate=gate,
        companion=companion,runtime_eligible=False,achievement_claimed=False,
        correction='3calendar-day blocks with missing-day masks; no new training, threshold, trade selection or parameters',source_hashes={n:sha(Path(__file__).with_name(n)) for n in ('mission_calendar_evaluation.py',Path(__file__).name)})
    with (experiment/'corrected_evaluation.json').open('x',encoding='utf-8') as f:json.dump(corrected,f,indent=2)
    rows=[]
    for name,r in (('Исходные правила',base),('CatBoost: ранний лидер',target)):
        rows.append(f"| {name} | {ratio(r['early'],r['leader_pairs'])} | {ratio(r['captured'],r['leader_pairs'])} | {ratio(r['unique_precision_n'],r['unique_precision_N'])} |")
    report=f"""# Обучение под назначение криптобота

История: {raw['days']:.0f} полных дней, 01.04–08.10.2026 включительно.
Полные внутридневные данные: {raw['population_complete']}/{raw['population_requested']} символов.
Глобальные дневные исходы: 699 запрошенных символов, включая исторически отмеченные.
Исторический PIT-universe не подтверждён. TEST ранее использовался в исследованиях.

Цель: глобальный Top20 с дневным оборотом ≥1 млн USDT, пересечение с watchlist;
локальный день 00:00–24:00 Europe/Budapest. Ранний захват: осталось ≥35% дневного роста.
Precision: первый BUY на уникальную пару день/символ. Эта цель отличается от старого Top15/22:00.

| TEST | Ранние захваты | Охват лидеров | Precision BUY |
|---|---:|---:|---:|
{chr(10).join(rows)}

TEST: {base['days']} общих наблюдаемых дней. Неполные дни исключены, а не признаны промахами.
Прирост ранних захватов: {target['early']-base['early']:+d}.
95% интервал при блоках из3календарных дней: {ci['early_count_delta95'] if ci else 'недостаточно дней'}.
Решение: **{gate['state']}**. Доказанного улучшения production не заявляется.
Сопровождение и удержание роста пока не прошли отдельный подтверждённый критерий.
Совпадений по всем16исходным полям входа: {strict}/{len(common)} общих подтверждённых эпизодов.
Сравнение только по времени/цене из исходного отчёта не используется для одобрения.

Реализованы отдельные головы: обнаружение лидера, ранний захват,
продолжение на1ч и повторный вход на4ч. Последние две обучены на реальных состояниях
контрольного replay и остаются диагностическими прогнозами без изменения SELL/BUY.
Chronological refit, mature-only labels, purged inner validation, frozen parameters.
Сравнение учитывает весь портфель из10позиций, замены, cooldown и защитные выходы.

Проверки: независимые raw-цены/дневные границы/ранги, доступность меток,
SHA моделей, совпадение прежнего контрольного replay и числители/знаменатели — PASS.
Исходный отчёт с3блоками по сохранившимся строкам сохранён; этот отчёт использует
зарегистрированную поправку на календарные пропуски. Торги и модели не менялись.

Следующий слой:35-дневный фиксированный forward shadow с immutable raw-данными,
as-of watchlist/universe, временем получения и выдачи, контролем запоздания и drift.
Fresh gate требует ≥30полных дней и ≥100лидер-дней, сравнения по целевой метрике,
подтверждённых acceptance clocks, полного покрытия и качества сопровождения.
Отсутствующие будущие исходы остаются pending; canary и production заблокированы.

Подробные результаты: result.json, corrected_evaluation.json,
independent_verification.json, registration.json, folds_*.json и dataset_*.npz.
"""
    with (experiment/'REPORT_RU.md').open('x',encoding='utf-8') as f:f.write(report)
    print(json.dumps(corrected),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);a=p.parse_args();publish(a.experiment)
