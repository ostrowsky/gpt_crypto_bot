"""Mission-only tables and leader accompaniment charts, with explicit cohorts."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from historical_signal_evaluation import sha

LABELS={'control':'Control','amplitude':'Amplitude','no_replacement':'Keep slots','combined':'Combined','impulse_catboost':'CatBoost entry','joint':'Direction + amplitude','exit_model':'Soft SELL hold'}


def table_rows(result):
    out=[]
    for arm,r in result['arms'].items():
        for cohort in ('full','test'):
            x=r[cohort];e=r['exits_'+cohort]
            out.append(dict(arm=arm,cohort=cohort,early=x['early'],leaders=x['leader_pairs'],captured=x['captured'],
                precision_n=x['precision_n'],precision_N=x['precision_N'],unique_precision_n=x['unique_precision_n'],unique_precision_N=x['unique_precision_N'],
                old_early=x['stored_early'],early_disagreements=x['corrected_early_disagreements'],
                confirmed_episodes=e['confirmed_n'],first_premature=e['first_early_exit_n'],retention=e['retention_mean'],
                accompaniment=e['accompaniment_mean'],any_remaining=e['any_remaining_n'],rebound_n=e['rebound_n'],rebound_N=e['rebound_N']))
    return out


def render(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as dates
    r=json.loads((output/'result.json').read_bytes());audit=json.loads((output/'independent_verification.json').read_bytes())
    if audit['status']!='PASS' or audit['result_sha256']!=sha(output/'result.json'):raise ValueError('unverified metrics')
    rows=table_rows(r)
    with (output/'comparison.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    arms=list(r['arms']);names=[LABELS[a] for a in arms];x=np.arange(len(arms));fig,axes=plt.subplots(3,1,figsize=(13,11),layout='constrained')
    for ax,key,title in zip(axes,('early','captured','precision_n'),('Early leaders / all daily leaders','Captured leaders / all daily leaders','Leader BUYs / eligible BUYs')):
        for offset,cohort,color in [(-.18,'full','#8caacb'),(.18,'test','#244b73')]:
            numer=[r['arms'][a][cohort][key] for a in arms]
            den=[r['arms'][a][cohort]['precision_N' if key=='precision_n' else 'leader_pairs'] for a in arms]
            vals=[100*n/d if d else np.nan for n,d in zip(numer,den)]
            bars=ax.bar(x+offset,vals,.35,color=color,label=cohort.upper())
            labels=[f'{n}\n/{d}' if key=='precision_n' else f'{n}/{d}' for n,d in zip(numer,den)]
            ax.bar_label(bars,labels=labels,fontsize=8,padding=3)
        ax.set_ylim(0,105);ax.set_xticks(x,names,fontsize=9);ax.set_ylabel('%');ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.legend()
    fig.suptitle('Leader mission re-audit | same 93-symbol population | corrected 22:00 CLOSE labels\n186 full days / 38 retrospective TEST days; no economic winner criterion')
    fig.savefig(output/'mission_comparison.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(13,11),layout='constrained')
    e=[r['arms'][a]['exits_test'] for a in arms]
    metrics=[([100*v['retention_mean'] if v['retention_mean'] is not None else np.nan for v in e],'First position retained rise / close peak through marker (%)'),
        ([100*v['accompaniment_mean'] if v['accompaniment_mean'] is not None else np.nan for v in e],'Time in leader positions before marker, INCLUDING re-entries (%)'),
        ([100*v['first_early_exit_n']/v['confirmed_n'] if v['confirmed_n'] else np.nan for v in e],'First position reduction BEFORE causal weakening marker (%)')]
    for ax,(vals,title) in zip(axes,metrics):
        bars=ax.bar(x,vals,color=['#244b73']+['#889eae']*5+['#ce7333'])
        ax.bar_label(bars,labels=[f"n={v['confirmed_n']}" for v in e],padding=3,fontsize=9)
        ax.set_title(title);ax.set_xticks(x,names,fontsize=9);ax.axhline(0,color='grey',lw=.7);ax.grid(axis='y',alpha=.2)
        low=min(0,min(v for v in vals if np.isfinite(v)));high=max(v for v in vals if np.isfinite(v));ax.set_ylim(low-2,high+max(5,abs(high)*.15))
    fig.suptitle('Leader accompaniment | TEST | confirmed 24-hour episodes only\nOperational weakening marker; early risk exits can be appropriate. Cohort means have different survivors.')
    fig.savefig(output/'exit_comparison.png',dpi=150);plt.close(fig)
    # Earliest same-entry TEST pairs where the frozen SELL timing differs, never selected by return.
    e0={(v['day'],v['symbol']):v for v in json.loads((output/'episodes_control.json').read_bytes())}
    e1={(v['day'],v['symbol']):v for v in json.loads((output/'episodes_exit_model.json').read_bytes())}
    reg=json.loads((output/'registration.json').read_bytes());selected=[]
    for pair in sorted(e0.keys()&e1.keys()):
        a,b=e0[pair],e1[pair]
        if a['entry_ts']<reg['test_boundary_ms'] or (a['entry_ts'],a['entry_price'])!=(b['entry_ts'],b['entry_price']):continue
        if a['exit_ts']!=b['exit_ts']:selected.append(pair)
    selected=selected[:2]
    if selected:
        fig,axes=plt.subplots(len(selected),1,figsize=(13,4*len(selected)),squeeze=False,layout='constrained')
        for ax,pair in zip(axes.ravel(),selected):
            a,b=e0[pair],e1[pair];p=Path(reg['market'])/'market'/(pair[1]+'_15m.json');raw=json.loads(p.read_bytes())
            begin=a['entry_ts']-4*3600000;end=a['entry_ts']+24*3600000
            data=[v for v in raw if begin<=v['t']+900000<=end]
            t=[datetime.fromtimestamp((v['t']+900000)/1000,timezone.utc) for v in data];ax.plot(t,[v['c'] for v in data],'k-',label='Actual closed15m price')
            for clock,color,label in [(a['entry_ts'],'#238b45','BUY'),(a['exit_ts'],'#cb3434','Control SELL'),(b['exit_ts'],'#da842e','Model SELL')]:
                ax.axvline(datetime.fromtimestamp(clock/1000,timezone.utc),color=color,ls='--',label=label)
            if a['state']=='CONFIRMED':ax.axvline(datetime.fromtimestamp(a['marker_ts']/1000,timezone.utc),color='#825cab',ls=':',label='Causal weakening marker')
            ax.set_title(f"{pair[1]} | {pair[0]} | same-entry first leader position | historical diagnostic")
            ax.xaxis.set_major_formatter(dates.DateFormatter('%m-%d %H:%M'));ax.set_ylabel('USDT');ax.grid(alpha=.2);ax.legend(fontsize=9,ncol=3)
        fig.savefig(output/'leader_exit_examples.png',dpi=150);plt.close(fig)
    lines=['# Повторная проверка по назначению бота','',
        'Основные критерии: раннее обнаружение, охват дневных лидеров, качество BUY и сопровождение до ослабления роста. Доходность и комиссии не определяют победителя.',
        'Все модели, пороги и решения сохранены. Пересчитаны метрики на максимальном прежнем периоде:186 дней,93/105 символов. TEST38 полных дней уже использовался ранее; это ретроспективная переоценка.',
        '', '## Исправление раннего захвата','',
        'Top15 и оставшееся движение теперь используют одинаковую цену закрытия на22:00 Europe/Budapest; время входа — фактический entry_ts решения, все цены на15m close. Старые аннотации зависели от1h/15m open clock и другого финального среза.',
        '**Контроль TEST:253 →247 ранних лидеров из570.** Охват и precision остались согласованы. На полном периоде1039 →1031/2790. Изменившиеся индивидуальные early-метки:32 TEST и158 full, часть изменений взаимно компенсируется.',
        'Это обнаруженный ручной TH-03/TH-05 FAIL прежних ранних аннотаций. Механический Harness PASS не устраняет этот дефект; новые метрики восстановлены из raw. Результаты старых экономических экспериментов не переписаны.',
        '', '## Входы: общий TEST','',
        '| Вариант | Ранние /570 | Охват /570 | Leader BUY / все BUY | Уникальные leader day-symbol / все BUY day-symbol | Решение по миссии |',
        '|---|---:|---:|---:|---:|---|']
    for arm in arms:
        a=r['arms'][arm]['test'];verdict='CONTROL' if arm=='control' else r['comparisons'][arm]['mission_verdict']
        lines.append(f"| {LABELS[arm]} | {a['early']}/570 | {a['captured']}/570 | {a['precision_n']}/{a['precision_N']} | {a['unique_precision_n']}/{a['unique_precision_N']} | {verdict} |")
    lines+=['', '## Гипотезы в исходном порядке','']
    c=r['capacity'];lines+=[f"1. **Capacity CatBoost:** {c['known']}/{c['issued']} известных конфликтов. Keep: лидеры {c['keep']['leader_n']}/{c['known']}, early {c['keep']['early_n']}/{c['known']}; модель: {c['model']['leader_n']}/{c['known']} и {c['model']['early_n']}/{c['known']}. {c['mission_verdict']}. Это повторные пары конфликтов на181-дневном источнике, не уникальный охват и не полный policy replay.",
        '2. **Amplitude / запрет замен / комбинация:** см. общий TEST выше; раздельные решения, прежние правила не подгонялись.',
        '3. **CatBoost impulse:** см. общий TEST; цель обучения75m return сохранена. Переоценка не является тестом новой модели, обученной раннему захвату.',
        '4. **Направление + амплитуда:** см. общий TEST; прежние вероятности и пороги сохранены.',
        '5. **Soft SELL hold:** отдельные общие пары и сопровождение ниже.',
        f"6. **Order flow execution:** {r['execution']['joined']}/{r['execution']['issued']} действий сопоставлены; отсрочек {r['execution']['actual_deferrals']}. Среди сопоставленных действий лидеров: {audit['joined_execution_leader_actions']}. Такой тест не подтверждает улучшение поиска/сопровождения лидеров."]
    lines+=['', '## Сопровождение лидеров: TEST','',
        'Операционное ослабление роста: после подъёма >=1 entryATR, два закрытия подряд >=2 currentATR ниже накопленного close-пика и со снижением EMA20. ATR14 — rolling mean true range. Горизонт наблюдения24ч. Это диагностическое определение, не знание будущей абсолютной вершины.',
        '| Вариант | Подтверждённые эпизоды | Первый выход до marker | Среднее удержание подъёма первой позицией | Средняя доля времени сопровождения с повторными входами | Любая позиция на marker | Новый максимум <=4ч после первого выхода |',
        '|---|---:|---:|---:|---:|---:|---:|']
    number=lambda v:'unknown' if v is None else f'{v:.4f}'
    for arm in arms:
        a=r['arms'][arm]['exits_test'];lines.append(f"| {LABELS[arm]} | {a['confirmed_n']} | {a['first_early_exit_n']}/{a['confirmed_n']} | {number(a['retention_mean'])} | {number(a['accompaniment_mean'])} | {a['any_remaining_n']}/{a['confirmed_n']} | {a['rebound_n']}/{a['rebound_N']} |")
    lines+=['','Retention — gross движение цены, не денежная доходность; отрицательные значения сохранены. Колонки имеют разные знаменатели. Последующие повторные входы включены только в сопровождение времени/наличие позиции; первый SELL сам по себе не означает полный отказ от монеты. Hard risk exits могут быть правильными до marker.',
        'Дополнительное время сопровождения добавлено после просмотра предварительной first-position диагностики: exploratory, без новой модели или подбора параметров.',
        '', '## Общие пары: исключение разного отбора','']
    for arm,c in r['comparisons'].items():
        e=c['exits'];lines.append(f"- {LABELS[arm]}: common {e['common_pairs']}, одинаковый вход+confirmed {e['confirmed_same_entry_n']}; Δretention {number(e['retention_delta_mean'])}, Δfirst-before-marker count {e['first_early_delta_n']}; familywise95%CI {e['retention_interval']}.")
        lines.append(f"  Entry familywise95% интервалы, п.п.: {c['entry']['intervals']}.")
    lines+=['', 'Поправка на24 сравнения, блоки по3 календарных дня,5000draws. Разный survivor состав может улучшить среднее сопровождение при ухудшении охвата; поэтому общие пары и таблица входов нужны одновременно.',
        '', '## Итог и ограничения','',
        'Переоценка отвечает на качество старых фиксированных гипотез относительно назначения бота. Она не доказывает бесполезность моделей, обученных непосредственно early-leader/trend-end целям. Следующий исследовательский приоритет — причинная диагностика преждевременного выхода/повторного входа в фактически найденных лидерах.',
        f"Независимая проверка PASS: {audit['daily_labels']} дневных меток, {audit['first_leader_entries']} first leader входов, {audit['scalar_episode_checks']} scalar эпизодов. Production не изменён.",
        'Данные availability-selected93/105; пропуски follow-up и неустановившиеся/неподтверждённые тренды разделены. Отсутствуют сертифицированные PIT-universe и live/receive-time parity. Новые метрики не заменяют требование свежей forward-проверки перед изменением поведения.',
        'Канонический Top15 относительный: в слабый день он может включать монеты с неположительным дневным изменением. Для них remaining positive move не определён; они не получают ранний захват. Это не обещание найти15 растущих монет каждый день.',
        'V1 завершился ошибкой индексации bootstrap; v2 не прошёл независимый scalar аудит на точной ATR-границе. Оба сохранены. V3 исправляет local-window ATR и численную границу1e-12*entry-price при прежних параметрах и решениях; v3 — итоговая версия.','',
        f"![Mission]({(output.resolve()/'mission_comparison.png').as_posix()})",'', f"![Accompaniment]({(output.resolve()/'exit_comparison.png').as_posix()})"]
    if selected:lines+=['',f"![Leader examples]({(output.resolve()/'leader_exit_examples.png').as_posix()})"]
    with (output/'report.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    files=['report.md','comparison.csv','mission_comparison.png','exit_comparison.png']+(['leader_exit_examples.png'] if selected else [])
    with (output/'report_receipt.json').open('x',encoding='utf-8') as f:json.dump(dict(result_sha256=sha(output/'result.json'),renderer_sha256=sha(__file__),files={p:sha(output/p) for p in files}),f,indent=2)
    print('RENDERED mission report',output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);render(p.parse_args().output)
