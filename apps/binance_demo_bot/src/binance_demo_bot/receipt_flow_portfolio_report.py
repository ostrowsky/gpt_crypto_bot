"""RFP09 post-replay diagnostics; never changes policies or promotes models."""
import argparse
from collections import Counter
import csv
from decimal import Decimal as D
import gzip
import json
from pathlib import Path
from . import receipt_flow_direction as io


def journal_reasons(events):
    reasons=Counter();orders=Counter();selected=fills=n=0
    for e in events:
        if e['evidence_kind']!='SIMULATION':raise ValueError('MIXED_EVIDENCE')
        kind=e['kind']
        if kind=='BUY_FILL':fills+=1
        if kind=='DECISION':
            choice=e['selected'];reason=e['reason']
            if (choice is None and not isinstance(reason,str)) or (choice is not None and reason is not None):
                raise ValueError('AMBIGUOUS_DECISION')
            n+=1;selected+=int(choice is not None);reasons['SELECTED' if choice is not None else reason]+=1
        elif kind in ('BUY_WAIT','SELL_WAIT','BUY_REJECTED_CAPACITY','BUY_EXPIRED','BUY_CANCELLED','RISK_LATCH'):
            orders[kind+':'+e.get('reason',kind)]+=1
    return dict(decision_N=n,selected_decisions_n=selected,buy_fills_n=fills,
        selection_fraction=selected/n if n else None,decision_reason_counts=dict(reasons),
        order_and_risk_reason_counts=dict(orders))


def _events(path):
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        for line in stream:yield json.loads(line)


def create_report(folder):
    folder=Path(folder).resolve()
    status=json.loads((folder/'status.json').read_text())
    if status['phase']!='COMPLETE_CONDITIONAL_SIMULATION':raise ValueError('INCOMPLETE_CAMPAIGN')
    report=io.read_frozen(folder/'report.json');audits=io.read_frozen(folder/'ledger-audits.json')
    plan=io.read_frozen(folder/'registration.json');signals=io.read_frozen(folder/'signals-manifest.json')
    if audits['state']!='PASS_LEDGER_ARTIFACT_AUDIT_ONLY' or audits['cases_n']!=len(report['cases']):raise ValueError('AUDIT_SCOPE')
    if io.file_sha(folder/'microprice-calibration.json')!=signals['microprice_calibration_sha256']:raise ValueError('BASELINE_CALIBRATION_HASH')
    rows=[];sources={str(folder/'report.json'):io.file_sha(folder/'report.json')}
    for c in report['cases']:
        out=folder/'cases'/c['case_id'];manifest=io.read_frozen(out/'manifest.json');result=io.read_frozen(out/'result.json')
        path=out/'events.jsonl.gz';sha=io.file_sha(path)
        if sha!=manifest['events_sha256'] or io.file_sha(out/'result.json')!=manifest['result_sha256']:
            raise ValueError('CASE_ARTIFACT_CHANGED')
        counts=journal_reasons(_events(path));sources[str(path)]=sha
        if counts['buy_fills_n']!=c['entries_n']:raise ValueError('DECISIONS_ARE_NOT_FILLS')
        net=c['net_interval_usdt'];fees=c['fees_usdt']
        rows.append(dict(c,**counts,net_plus_fees_diagnostic_usdt=str(D(net)+D(fees)) if net is not None else None))
    cost=[r for r in rows if r['admission']=='cost-aware']
    complete=[r for r in cost if r['test_complete_days_n']==r['test_requested_days_N']==2]
    counts=dict(cost_aware_cases_N=len(cost),complete_test_cases_n=len(complete),
        positive_test_mean_n=sum(D(r['test_mean_daily_net_pnl_usdt'])>0 for r in complete),
        zero_test_mean_n=sum(D(r['test_mean_daily_net_pnl_usdt'])==0 for r in complete),
        negative_test_mean_n=sum(D(r['test_mean_daily_net_pnl_usdt'])<0 for r in complete),
        incomplete_test_cases_n=len(cost)-len(complete))
    supplemental=dict(state='COMPLETE_DIAGNOSTIC_REPORT_ONLY',cases=rows,counts=counts,
        inputs=sources,reporter_source_sha256=io.file_sha(__file__),financial_certification=False,promotion_allowed=False)
    io.freeze(folder/'decision-analysis.json',supplemental)
    columns=['case_id','net_interval_usdt','test_boundary_net_diagnostic_usdt','test_mean_daily_net_pnl_usdt',
        'test_complete_days_n','test_requested_days_N','entries_n','closed_cycles_n','fees_usdt',
        'net_plus_fees_diagnostic_usdt','unmarked_held_ticks_n','decision_N','selected_decisions_n','buy_fills_n']
    with (folder/'comparison.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    def fmt(v):return 'неизвестно' if v is None else f'{D(v):.6f}'
    lines=['# Портфельная проверка microprice и CatBoost','',
        'Условная симуляция RFP-1: BTC/ETH/SOL, 9–12 ноября 2026 Europe/Budapest; TEST — 11–12 ноября. На каждый вариант один общий портфель 100 USDT и одна позиция до 30 USDT. Деньги и комиссии учитываются через Decimal; позиции и баланс переносятся через полночь.','',
        'База: комиссия 0,10% и adverse slippage 0,05% на сторону, задержка 1 секунда. Стресс: 0,20%/0,10% и 2 секунды. Это заранее заданные предположения симуляции, не фактические комиссии или исполнения аккаунта.','',
        'Проверены все 60 вариантов и CASH. P/B/F — фиксированные CatBoost цена/стакан/сделки; M0 — исходный microprice; M1 — линейная оценка доходности по microprice, обученная только на TRAIN. Подбора порога или переобучения CatBoost нет.','',
        '## Вход только при прогнозируемом преимуществе над издержками: база','',
        '| Метод | Горизонт, сек | Net OOS, USDT (диагностика границ) | TEST средний полный день | Полных TEST-дней / 2 | Входов / выбранных сигналов | Комиссии OOS |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['admission']=='cost-aware' and r['profile']=='base':
            lines.append(f"| {r['arm']} | {r['horizon_seconds']} | {fmt(r['net_interval_usdt'])} | {fmt(r['test_mean_daily_net_pnl_usdt'])} | {r['test_complete_days_n']}/2 | {r['entries_n']} / {r['selected_decisions_n']} | {fmt(r['fees_usdt'])} |")
    lines+=['','CASH: net 0, средний полный день 0, без сделок. Значения Net OOS не заменяют целевую метрику, если промежуточные оценки открытой позиции неизвестны. TEST наследует капитал и позиции с 9–10 ноября.','',
        '## Вход по положительному направлению: база','',
        '| Метод | Горизонт, сек | Net OOS после комиссий | Net + комиссии (спред/slippage остаются) | Полных TEST-дней / 2 | Закрытых циклов | Неизвестных clocks с позицией |','|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['admission']=='direction' and r['profile']=='base':
            lines.append(f"| {r['arm']} | {r['horizon_seconds']} | {fmt(r['net_interval_usdt'])} | {fmt(r['net_plus_fees_diagnostic_usdt'])} | {r['test_complete_days_n']}/2 | {r['closed_cycles_n']} | {r['unmarked_held_ticks_n']} |")
    lines+=['','## Полнота и причины отказов','',
        f"Сценариев cost-aware: {counts['cost_aware_cases_N']}. Оба полных TEST-дня известны у {counts['complete_test_cases_n']}; из них положительный средний PnL у {counts['positive_test_mean_n']}, нулевой у {counts['zero_test_mean_n']}, отрицательный у {counts['negative_test_mean_n']}. Неполные: {counts['incomplete_test_cases_n']}. Это связанные варианты одного эксперимента, не независимые повторения.",
        '', 'Каждое решение содержит ссылку на замороженную строку признаков. SELECTED означает решение выбрать вход, а не исполнение. Отказы, отсутствие сигнала, риск, cooldown, истечения TTL и недостаточная ликвидность разделены в decision-analysis.json; знаменатель каждого selection_fraction — decision_N, пустой знаменатель null.','',
        'День с неизвестной оценкой удерживаемой позиции исключается из среднего, а не превращается в нулевой/выигрышный. Прошлая граница не восстанавливается будущей ценой. Учитываются открытая экспозиция, незавершённые выходы, комиссии и наблюдаемая просадка.','',
        '## Ограничения','',
        'Это два уже раскрытых TEST-дня и три ликвидные Demo-пары. Исполнение — proxy по сохранённому best bid/ask, ограниченный объёмом и свежестью receipt; биржевой возраст котировки, очередь и исторические LOT_SIZE/tick-фильтры не сертифицированы. Есть источник сырого снимка для каждого исполнения и оценки.','',
        f"Полный Truth Harness во время кампании: {plan['truth_harness']['status'].upper()}. Сохраняется TH-11 старого канонического отчёта. Независимая сверка ledger всех {len(rows)} сценариев — PASS_LEDGER_ARTIFACT_AUDIT_ONLY, это не финансовая сертификация.",
        '', 'Реальных новых ордеров нет. Симуляции не передавались как ACTUAL_DEMO в обучение, модели и лимиты работающего бота не изменены.','',
        '[Все таблицы и первые хронологические сделки](comparison.html) · [CSV](comparison.csv) · [Причины решений](decision-analysis.json) · [Дневной график](plots/daily-pnl.png)']
    (folder/'findings-ru.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return dict(state=supplemental['state'],cases_n=len(rows),counts=counts,
                report=str(folder/'findings-ru.md'),promotion_allowed=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--folder',required=True)
    args=parser.parse_args()
    from .public_flow_dataset import own_runtime_path
    folder=own_runtime_path(Path(__file__).resolve().parents[2],args.folder)
    print(json.dumps(create_report(folder),ensure_ascii=True),flush=True)


if __name__=='__main__':main()
