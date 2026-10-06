"""Verified mission report, exit causes and chronological same-entry price examples."""
from __future__ import annotations
import argparse,json,csv
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from historical_signal_evaluation import sha
from run_turnover_economics import publish
from leader_trend_continuation import category

ENTRY_FIELDS=('sym','tf','mode','entry_ts','entry_price','entry_i','trail_k','max_hold_bars',
    'entry_score','ranker_final_score','ranker_top_gainer_prob','top_gainer_score','allocation_score',
    'entry_rsi','entry_daily_range','entry_intraday_change_pct')


def verify_entry_identity(control,trend,boundary):
    a={(e['day'],e['symbol']):e for e in control};b={(e['day'],e['symbol']):e for e in trend};n=0
    for key in a.keys()&b.keys():
        x,y=a[key]['trade'],b[key]['trade']
        if x['entry_ts']<boundary or (x['entry_ts'],x['entry_price'])!=(y['entry_ts'],y['entry_price']):continue
        if any(x[k]!=y[k] for k in ENTRY_FIELDS):raise ValueError('TEST same clock/price but different immutable entry state')
        n+=1
    return n


def table_rows(result):
    return [dict(arm=k,cohort=cohort,early=r[cohort]['early'],leaders=r[cohort]['leader_pairs'],
        captured=r[cohort]['captured'],leader_BUY=r[cohort]['precision_n'],all_BUY=r[cohort]['precision_N'],
        confirmed=r['exits_'+cohort]['confirmed_n'],retained=r['exits_'+cohort]['retention_mean'],
        held_time=r['exits_'+cohort]['accompaniment_mean'],first_before=r['exits_'+cohort]['first_early_exit_n'])
        for k,r in result['arms'].items() for cohort in ('full','test')]


def render(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as dates
    r=json.loads((directory/'result.json').read_bytes());audit=json.loads((directory/'independent_verification.json').read_bytes())
    if audit['status']!='PASS' or audit['result_sha256']!=sha(directory/'result.json'):raise ValueError('unverified result')
    reg=json.loads((directory/'registration.json').read_bytes())
    identity_n=verify_entry_identity(json.loads((directory/'leader_entries_control.json').read_bytes()),
        json.loads((directory/'leader_entries_trend.json').read_bytes()),reg['test_boundary_ms'])
    rows=table_rows(r)
    with (directory/'comparison.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    fig,axes=plt.subplots(2,3,figsize=(15,9),layout='constrained')
    for ax,(key,title) in zip(axes[0],(('early','Early leaders'),('captured','Captured leaders'),('precision_n','Leader BUY precision'))):
        for offset,cohort,color in ((-.18,'full','#95b3c7'),(.18,'test','#285578')):
            x=[r['arms'][k][cohort] for k in ('control','trend')];nums=[a[key] for a in x];dens=[a['precision_N' if key=='precision_n' else 'leader_pairs'] for a in x]
            bars=ax.bar(np.arange(2)+offset,[100*n/d if d else np.nan for n,d in zip(nums,dens)],.35,color=color,label=cohort.upper())
            ax.bar_label(bars,labels=[f'{n}/{d}' for n,d in zip(nums,dens)],padding=3,fontsize=9)
        ax.set_ylim(0,105);ax.set_title(title);ax.set_ylabel('%');ax.set_xticks([0,1],['Control','Trend hold']);ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    e=[r['arms'][k]['exits_test'] for k in ('control','trend')]
    for ax,(key,title,denominator) in zip(axes[1],(('retention_mean','First-position retained peak rise','ratio'),
        ('accompaniment_mean','Held time INCLUDING re-entries','ratio'),('first_early_exit_n','First reduction before weakening','count'))):
        vals=[100*a[key] if denominator=='ratio' else 100*a[key]/a['confirmed_n'] for a in e]
        bars=ax.bar([0,1],vals,color=['#285578','#da853a']);ax.bar_label(bars,labels=[f'{v:.2f}%\nn={a["confirmed_n"]}' for v,a in zip(vals,e)],padding=3)
        ax.set_ylim(min(0,min(vals))-5,max(vals)+15);ax.set_title(title);ax.set_xticks([0,1],['Control','Trend hold']);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Fixed causal trend-confirmed soft SELL | 186 days, 93 symbols | exposed retrospective TEST\nBottom row: different survivor cohorts; causal comparison uses exact same entries')
    fig.savefig(directory/'comparison.png',dpi=150);plt.close(fig)
    groups=r['causes']['control']['test']['categories'];keys=list(groups)
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained');bars=ax.bar(keys,[groups[k]['before'] for k in keys],color='#285578')
    ax.bar_label(bars,labels=[f"{groups[k]['before']}/{groups[k]['n']}" for k in keys],padding=3);ax.set_ylim(0,max(groups[k]['before'] for k in keys)+30)
    ax.set_ylabel('Leader first positions');ax.set_title('Control TEST: first reduction BEFORE operational weakening, by final SELL category\nCounts are diagnostic; early risk exits can be appropriate; partial exits classified by final reason')
    ax.grid(axis='y',alpha=.2);fig.savefig(directory/'exit_causes.png',dpi=150);plt.close(fig)
    e0,e1=({(e['day'],e['symbol']):e for e in json.loads((directory/f'episodes_{k}.json').read_bytes())} for k in ('control','trend'))
    reg=json.loads((directory/'registration.json').read_bytes())
    selected=sorted([p for p in e0.keys()&e1.keys() if e0[p]['entry_ts']>=reg['test_boundary_ms'] and
        (e0[p]['entry_ts'],e0[p]['entry_price'])==(e1[p]['entry_ts'],e1[p]['entry_price']) and e0[p]['exit_ts']!=e1[p]['exit_ts']],key=lambda p:(e0[p]['entry_ts'],p))[:3]
    if selected:
        fig,axes=plt.subplots(len(selected),1,figsize=(13,4*len(selected)),squeeze=False,layout='constrained')
        for ax,p in zip(axes.ravel(),selected):
            a,b=e0[p],e1[p];raw=json.loads((Path(reg['market'])/'market'/(p[1]+'_15m.json')).read_bytes())
            data=[v for v in raw if a['entry_ts']-4*3600000<=v['t']+900000<=a['entry_ts']+24*3600000]
            ax.plot([datetime.fromtimestamp((v['t']+900000)/1000,timezone.utc) for v in data],[v['c'] for v in data],'k-',lw=1,label='Actual closed15m price')
            for at,color,label in ((a['entry_ts'],'#238b45','BUY'),(a['exit_ts'],'#c23b3b','Control SELL'),(b['exit_ts'],'#da853a','Trend hold SELL')):
                ax.axvline(datetime.fromtimestamp(at/1000,timezone.utc),color=color,ls='--',label=label)
            if a['state']=='CONFIRMED':ax.axvline(datetime.fromtimestamp(a['marker_ts']/1000,timezone.utc),color='#825cab',ls=':',label='Operational weakening')
            ax.set_title(f'{p[1]} | {p[0]} | earliest changed same-entry TEST case | historical, not forecast\nControl: {category(a["exit_reason"])}; rule: {category(b["exit_reason"])}',fontsize=10)
            ax.set_ylabel('USDT');ax.xaxis.set_major_formatter(dates.DateFormatter('%m-%d %H:%M'));ax.grid(alpha=.2);ax.legend(ncol=4,fontsize=8)
        fig.savefig(directory/'leader_examples.png',dpi=150);plt.close(fig)
    c=r['comparison'];diag=r['causes']['control']['test'];lines=['# Проверка сопровождения лидеров','',
        'Одна заранее зафиксированная гипотеза: отложить только первый мягкий SELL при подтверждённом росте EMA20 и цене в пределах1ATR от уже известного close-максимума. Подъём>=1entryATR; предел60мин, жёсткие выходы приоритетны.',
        'Максимальный архив186.333дня,186полных дней,93/105монет. TEST38дней ранее раскрыт; ретроспективная проверка, не свежий holdout. Реальные production правила не изменены.',
        '', '| Вариант | Период | Ранние / лидеры | Охват / лидеры | Leader BUY / все BUY | Подтверждённые эпизоды | Удержание подъёма | Время сопровождения с re-entry | Первый выход до marker |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:lines.append(f"|{row['arm']}|{row['cohort']}|{row['early']}/{row['leaders']}|{row['captured']}/{row['leaders']}|{row['leader_BUY']}/{row['all_BUY']}|{row['confirmed']}|{row['retained']:.4f}|{row['held_time']:.4f}|{row['first_before']}/{row['confirmed']}|")
    lines+=['','## Диагностика исходных выходов TEST','', '| Причина | Первый выход до marker / эпизоды категории | Последующий новый максимум / известные |','|---|---:|---:|']
    for k,g in diag['categories'].items():lines.append(f"|{k}|{g['before']}/{g['n']}|{g['rebound_n']}/{g['rebound_N']}|")
    lines+=['',f"Фактические блокировки cooldown внутри подтверждённых лидерских эпизодов: {diag['cooldown_events_in_confirmed_leader_episodes']}, уникальные day-symbol эпизоды {diag['cooldown_unique_episode_pairs']}/{diag['confirmed_n']}. Один blocked candidate может попадать в несколько перекрывающихся эпизодов; event count дедуплицирован. Это не обещание успешного альтернативного BUY.",
        '', '## Сопоставимые входы и решение','',
        f"Одинаковые входы+confirmed: {c['accompaniment']['n']}; дней {c['accompaniment']['days']}. Δвремени сопровождения: {c['accompaniment']['mean']}; familywise95%CI {c['accompaniment']['interval']}.",
        f"Дополнительно проверены все {identity_n} TEST пары с одинаковыми entry clock/price: совпадают timeframe, mode, ATR trail multiplier, max-hold, индексы и все неизменяемые признаки входа. Будущие extrema/exit/capture поля в сопоставление не входят.",
        f"Δудержания подъёма: {c['exits']['retention_delta_mean']}; CI {c['exits']['retention_interval']}; Δчисла ранних первых выходов {c['exits']['first_early_delta_n']}.",
        f"Интервалы раннего захвата/охвата/precision, п.п.: {c['entry']['intervals']}.",
        f"Решение: **{r['verdict']['status']}**. Проверки: `{json.dumps(r['verdict']['checks'])}`.",
        f"Все первые soft решения: {r['policy_counts']['first_soft_decisions']}, отсрочки {r['policy_counts']['deferrals']}, active ticks {r['policy_counts']['active_ticks']}.",
        f"Из active trace: {audit['causal_active_ticks']} причинных тиков и {audit['terminal_legacy_rechecks']} отдельный legacy boundary recheck. Последний не считается реальным решением сопровождения; принудительные закрытия исключены из качества выходов.",
        '', '## Ограничения и проверка','',
        'Операционный marker: после подъёма>=1entryATR два закрытия подряд>=2currentATR ниже running close peak со снижением EMA20. Это наблюдаемое ослабление, не абсолютная будущая вершина; защитные выходы до него могут быть верны. Отдельные survivor средние не доказывают улучшение. UNKNOWN follow-up не считается нулём.',
        'Старые ранние аннотации имеют TH03/05 FAIL временного выравнивания; все основные метрики пересчитаны из raw00:00-open/22:00-close. Контроль заново воспроизведён с точным совпадением всех полей прежних сделок.',
        f"Независимая проверка **PASS**: {audit['scalar_episode_checks']} scalar эпизодов, {audit['first_soft_decisions']} soft решений, {audit['active_ticks']} active ticks, {audit['actual_cooldown_candidates']} cooldown кандидатов. Нативные индикаторы стратегии унаследованы по source/input SHA; EMA/ATR нового правила независимо восстановлены.",
        'Первый независимый аудит FAIL на backdated boundary callback сохранён. Повторный аудит допускает только точный len(data)-2 callback у open_at_end с последним clock=end; реальные обратные clocks и reactivation запрещены. Первоначальный replay завершился нормально; восстановление не понадобилось. Параметры и сделки не менялись.',
        'Изменённый SELL на графике может быть вызван заменой позиции из-за другого состава портфеля, а не прямой отсрочкой WEAK. Например, ENAUSDT закрыта новой политикой раньше из-за REPLACEMENT. Общие пары измеряют полный эффект политики, включая доступность слотов.',
        'Account/DD/costs сохранены в result.json как безопасность; они не определяют winner миссии. PIT-universe/live receive/fill parity не сертифицированы; любое ретроспективное улучшение требует новой forward shadow проверки.',
        '',f"![Сравнение]({(directory/'comparison.png').resolve().as_posix()})",'',f"![Причины]({(directory/'exit_causes.png').resolve().as_posix()})"]
    if selected:lines+=['',f"![Примеры]({(directory/'leader_examples.png').resolve().as_posix()})"]
    (directory/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    outputs=['comparison.csv','comparison.png','exit_causes.png','report.md']+(['leader_examples.png'] if selected else [])
    publish(directory/'report_receipt.json',dict(result_sha256=sha(directory/'result.json'),renderer_sha256=sha(Path(__file__)),immutable_entry_identity_checked=identity_n,files={n:sha(directory/n) for n in outputs}))
    print('Rendered verified report',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);render(p.parse_args().run)
