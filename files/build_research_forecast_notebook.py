"""Build a standalone, clean-output interview demo from reviewed research code."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "research" / "Binance_BTC_ETH_SOL_ML_Researcher_Demo_v3.ipynb"


def build_research():
    cells = []
    def md(text):
        cells.append(dict(cell_type="markdown", metadata={}, source=text.strip().splitlines(keepends=True)))
    def code(text, hidden=False):
        cells.append(dict(cell_type="code", execution_count=None, outputs=[],
                          metadata={"jupyter": {"source_hidden": True}} if hidden else {},
                          source=text.strip().splitlines(keepends=True)))
    md("""
# ML Researcher: от сырых свечей до проверяемого прогноза и API

**Демо для интервью · BTCUSDT / ETHUSDT / SOLUSDT · версия 3.1**

Задача — прогнозировать **всю траекторию на 1…15 минут** по минутным свечам
Binance Spot, сравнить модели с persistence и показать инженерный путь до
запускаемого сервиса. Это демонстрация исследования и сервиса прогнозирования;
качество модели доказывается экспериментом, доходность торговли здесь не измеряется.

Внутри — ответы на **все 10 вопросов вакансии**, код, фактические метрики после
запуска, проверки причинности и отдельный блок запуска FastAPI.

**Быстрый старт:** отдельный Python 3.11 / Jupyter kernel → выполнить ячейку
установки ниже → **Restart Kernel → Run All**. Internet нужен для загрузки
публичных свечей, API-ключ Binance не нужен. Кэш делает повторный запуск на тех
же данных воспроизводимым. Код встроен: этот `.ipynb` не требует репозитория бота.
Сервис запускается отдельно в последней секции; Run All не открывает порт.

**Что доказано, а что нет:** исходный v2 был выполнен частично, содержал
предупреждения о несходимости, но не содержал завершённой итоговой таблицы.
Его картинки не служат доказательством превосходства моделей. В этом демо
выводы и ответы о числах строятся исключительно по новой таблице.
""")
    md("""
## Исправления методологии

| Риск в v2 | Исправление |
|---|---|
| `now()` отдельно для каждого актива, незакрытая последняя свеча | Общий фиксированный UTC cutoff; только закрытые бары; hash исходных данных |
| Пропуски только печатались, `dropna` мог сжать временную ось | Точная минутная сетка; конфликтующие дубли/пропуски/плохая схема останавливают запуск |
| Validation одновременно выбирала эпоху LSTM и ширину интервала | Отдельные train → tune → calibration → test; purge по времени доступности последней метки |
| 24 случайно разнесённых origin без оценки устойчивости | Предзаданный часовой UTC grid; полные test-дни; paired bootstrap по дням |
| «Лучший live» выбирался по test, persistence исключалась | Выбор только по tune, включая persistence; test не меняет выбор и веса |
| Усреднение MAE в USDT по BTC/ETH/SOL | Сопоставление по ошибке log return и относительному приросту |
| h15-интервал в цене, остальные горизонты отсутствовали | Калибровка каждого горизонта в log-return space; coverage с числителем и знаменателем, interval score |
| Несходимость и ошибки тяжёлых моделей могли скрываться | Масштабирование классического fit по доступной истории; явные COMPLETE / FAIL / UNAVAILABLE |
| TFT зависел от индекса строк и терял крайние decoder-окна | UTC-границы, сохранённый time_idx, causal encoder, будущий decoder без реальных будущих OHLCV |
| Частичная выгрузка принималась как успешная | Полнота периода обязательна; ограниченные retries; незавершённый кэш не сохраняется |

В v2 **уже были полезные меры**: хронологический split, gap, train-only scaler,
регрессоры SARIMAX без будущего объёма. Они сохранены и усилены проверками.
Перекрытие *исторического контекста* между split само по себе допустимо;
перекрытие будущих меток при обучении и настройке — недопустимо.
""")
    md("""
## 0. Окружение и зафиксированный протокол

Установка меняет только выбранное вами окружение ноутбука. Используйте отдельный
kernel. Дополнительные LSTM/TFT/Prophet запускаются по явному выбору до эксперимента;
их отсутствие не выдаётся за выполненный эксперимент.
""")
    requirements = (ROOT / "research" / "requirements-forecast.txt").read_text(encoding="utf-8")
    packages = [line for line in requirements.splitlines() if line and not line.startswith("#")]
    code("# В отдельном Jupyter kernel выполните один раз, затем перезапустите kernel:\n# %pip install " + " ".join(packages)
         + "\n# Для опциональных моделей (отдельный эксперимент с зафиксированными версиями):"
         + "\n# %pip install torch prophet lightning pytorch-forecasting\n"
         + "import importlib.util\n"
         + "required = ['numpy', 'pandas', 'scipy', 'sklearn', 'xgboost', 'statsmodels', 'matplotlib', 'ipywidgets']\n"
         + "missing = [p for p in required if importlib.util.find_spec(p) is None]\n"
         + "if missing:\n    raise RuntimeError('Установите зависимости из команды выше: ' + ', '.join(missing))")
    source = (ROOT / "files" / "research_forecast.py").read_text(encoding="utf-8")
    source = source.rsplit('\nif __name__ == "__main__":', 1)[0]
    code(source, hidden=True)
    code(f"SOURCE_SHA256 = {hashlib.sha256((ROOT/'files'/'research_forecast.py').read_text(encoding='utf-8').encode('utf-8')).hexdigest()!r}\n"
         + "CFG = ForecastConfig()\n"
         + "# End exclusive: [2026-05-04 00:00 UTC, 2026-09-01 00:00 UTC).\n"
         + "# Для НОВОГО эксперимента измените дату ДО просмотра результатов.\n"
         + "# 60 train / 15 tune / 15 calibration / 30 sealed test days.\n"
         + "# CFG = ForecastConfig(models=('Persistence','Ridge','XGBoost','ARIMA','SARIMA','SARIMAX','LSTM','TFT'), classical_window=2880)\n"
         + "CACHE = Path('forecast_demo_artifacts/cache')\n"
         + "EVIDENCE = Path('forecast_demo_artifacts/benchmark.json')\n"
         + "print(json.dumps(asdict(CFG), ensure_ascii=False, indent=2))")
    md("""
## 1. Сильный проект по forecasting — задача, данные, baseline, результат

**Ответ:** «Мой пример — исследование краткосрочного прогноза BTC, ETH и SOL:
минутные OHLCV Binance, горизонт 1–15 минут, цель — накопленная лог-доходность
от последнего закрытого бара. Baseline — неизменность цены. Сравниваю Ridge,
XGBoost, ARIMA, SARIMA и SARIMAX; DL — отдельные явно включаемые гипотезы.
Результат оцениваю по ошибке на одинаковых будущих timestamp, а не по похожести
графиков. Числа привожу из таблицы ниже; стабильное превосходство заранее не обещаю».

Использую **120 дней**, минимальный объём для выбранного протокола:
60 train + 15 tune + 15 calibration + 30 test. Это не универсальная нижняя
граница объёма данных для forecasting. Длинная test-часть нужна для проверки
устойчивости, а не потому, что каждый прогноз должен помнить четыре месяца.
Финальный период — август 2026; сентябрь, уже просмотренный в предыдущем
30-дневном демо, исключён. Архивы Binance проверяются по SHA256 CHECKSUM;
микросекунды spot-архивов явно переводятся в миллисекунды.
[Формат и контрольные суммы Binance](https://github.com/binance/binance-public-data).
В этом запуске торговые правила не меняются.
""")
    code("market, input_manifest = {}, {}\nfor symbol in CFG.symbols:\n"
         + "    market[symbol], input_manifest[symbol] = fetch_history(symbol, CFG, CACHE)\n"
         + "    print(symbol, len(market[symbol]), input_manifest[symbol]['sha256'])\n"
         + "display(pd.DataFrame([{ 'symbol':s, 'rows':len(d), 'start':d.open_time.iloc[0], 'end':d.open_time.iloc[-1], 'missing_minutes':0 } for s,d in market.items()]))")
    md("""
## 2. Честный backtest и доступность данных

**Ответ:** «Разбиваю данные по времени: 60 дней train, 15 tune, 15 calibration,
30 untouched test. Train обучает модель и scaler; tune выбирает модель/эпоху;
calibration задаёт интервалы; test оценивает уже зафиксированное решение.
Удаляю origin, если его последняя будущая метка ещё не была доступна до начала
следующей части. Внешние данные присоединяю по available_at, а не только event_time.
В этом ноутбуке устранены выбор модели по test и двойное использование validation».

Прогноз выдаётся после закрытия бара: `available_at = open_time + 1 минута + 2 секунды`.
Две секунды — **явное допущение о задержке**, а не наблюдённые исторические времена
доставки. Binance historical klines не позволяют доказать точную point-in-time
доступность прошлого; это ограничение записывается в каждый запуск.

ARIMA(1,1,0) оценивает AR-коэффициент методом Юла–Уокера на разностях;
это аналитическое оценивание, а не успешная «сходимость» итерационного MLE.
[API statsmodels](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.fit.html).

Фиксированные ML/SARIMA/SARIMAX и rolling refit ARIMA — разные
предзаданные политики обновления. Таблица сравнивает именно эти политики;
она не доказывает превосходство одной архитектуры при равном compute budget.
""")
    code("prepared_preview = {s:prepare(market[s],CFG) for s in CFG.symbols}\n"
         + "split_preview = {s:split_frames(d,CFG)[0] for s,d in prepared_preview.items()}\n"
         + "display(pd.DataFrame([{ 'symbol':s, 'split':name, 'n_origins':len(d), 'first_available':d.available_at.min(), 'last_label_available':d.label_available_at.max() } for s,sp in split_preview.items() for name,d in sp.items()]))\n"
         + "for s,sp in split_preview.items():\n"
         + "    for a,b in zip(('train','tune','calibration'),('tune','calibration','test')):\n"
         + "        assert sp[a].label_available_at.max() < sp[b].available_at.min()\n"
         + "print('PASS: метки предыдущего split известны до следующего split')")
    md("""
## 3. Новая модель показывает +7%: как проверить реальный прирост

**Ответ:** «Сначала уточняю метрику: например, снижение MAE на 7%, а не рост
доходности. Фиксирую параметры и основной горизонт до проверки. Сравниваю
парные ошибки на тех же origin, затем — на нескольких временных окнах и активах.
Из-за зависимости наблюдений считаю bootstrap по временным блокам, а не по
отдельным минутам. Проверяю ablation признаков, устойчивость к режимам и масштабу
данных. Если перебирал много вариантов, учитываю multiple testing и подтверждаю
результат на новых forward-данных. Для торговли отдельно учитываю costs и portfolio».

В демо предусмотрены три expanding-window fold **только внутри train**, регуляризация
Ridge/XGBoost, отдельный tune, а для LSTM — выбор checkpoint по tune и train-only
нормализация targets. Финальный test не участвует в поиске гиперпараметров.
Задан минимум **30 полных test-дней**. Для семейства сравнений применена
Bonferroni-поправка; показаны обычный и familywise CI и чувствительность к
блокам из трёх последовательных дней. Интервал, пересекающий ноль, —
`INCONCLUSIVE`, отрицательный целиком — `NO_IMPROVEMENT`, положительный
целиком — `SUPPORTED_DIAGNOSTIC`. Больше данных устраняет прежнюю нехватку
test-дней, но не гарантирует значимости. Число bootstrap-повторов — 10 000.
Обучение Ridge/XGBoost идёт на фиксированном 15-минутном grid (не по результатам);
SARIMA/SARIMAX оцениваются по последним 2880 минутам train и далее не переобучаются.
""")
    code("experiment = run_experiment(market, CFG)\n"
         + "save_evidence(experiment, input_manifest, EVIDENCE)\n"
         + "benchmark = pd.DataFrame(experiment['results'])\n"
         + "display(benchmark[['symbol','model','n_origins','MAE_h15_return','MAE_h15_USDT','improvement_pct','selected_before_test','n_time_blocks','verdict']])\n"
         + "display(pd.DataFrame(experiment['status']))\n"
         + "print('Параметры/версии/input hashes:', EVIDENCE.resolve())")
    code("cv_results, cv_traces = expanding_window_check(experiment, CFG, model_names=tuple(n for n in CFG.models if n != 'Persistence'), return_traces=True)\n"
         + "display(pd.DataFrame(cv_results)[['symbol','model','fold','n_train','n_origins','improvement_pct']])")
    md("""
## 4. Простые и сложные подходы

**Ответ:** «В этом проекте основной воспроизводимый benchmark включает persistence,
линейную Ridge, boosting XGBoost, ARIMA, SARIMA и SARIMAX. Prophet, LSTM и TFT
подготовлены как опциональные эксперименты. SARIMAX использует только известный
календарь: реальные будущие объёмы и цены в exog не передаются. Период 60 минут у SARIMA —
гипотеза, не установленный факт сезонности. Сложность принимаю только после
устойчивого OOS-прироста. ETS или отдельную state-space разработку как свой
подтверждённый опыт по этому ноутбуку не заявляю».

Опциональные DL-адаптеры не считаются проверенными только потому, что код написан.
Результат `UNAVAILABLE` или `FAIL` виден явно. TFT здесь обучается отдельно для
каждого актива: это исключает случайное несоответствие временных границ panel.
После каждого decoder окна проверяется точный timestamp. Prophet с суточной
сезонностью требует минимум 2880 минут истории на fit; 12 часов недостаточно.

SARIMA(1,1,0)×(1,0,0,60) и SARIMAX оцениваются **conditional least squares**
на train. Для AR-only остатков полный прогноз строится причинной рекурсией;
успешная остановка SciPy optimizer проверяется. Это явный estimator, его
нельзя называть успешной MLE-сходимостью statsmodels. ARIMA использует
аналитический Юла–Уокера на последнем причинном двухдневном окне.
""")
    code("comparison = benchmark.copy()\n"
         + "comparison['RMSE_h15_return'] = comparison.RMSE_by_horizon.map(lambda x:x[-1])\n"
         + "comparison['PI90_coverage_h15'] = comparison.PI90_coverage_by_horizon.map(lambda x:x[-1])\n"
         + "comparison['PI90_covered_h15'] = comparison.PI90_covered_by_horizon.map(lambda x:x[-1])\n"
         + "display(comparison[['symbol','model','n_origins','MAE_h15_return','RMSE_h15_return','MAE_h15_USDT','improvement_pct','ci95_familywise_improvement_pct','verdict','direction_population_correct','direction_population_n','direction_hit_rate','direction_balanced_accuracy','direction_gain_pp','direction_ci95_familywise_gain_pp','direction_verdict','PI90_covered_h15','PI90_denominator','PI90_coverage_h15']])\n"
         + "display(comparable_ranking(experiment['results'], CFG))\n"
         + "display(pd.DataFrame(experiment['direction_baselines']))\n"
         + "display(pd.DataFrame([dict(symbol=r['symbol'],model=r['model'],**p) for r in experiment['results'] for p in r['test_subperiods']]))\n"
         + "plot_comparison(experiment,CFG)\n"
         + "for symbol in CFG.symbols:\n"
         + "    ranked=comparison[comparison.symbol==symbol].sort_values('MAE_h15_return')\n"
         + "    best=ranked.iloc[0]\n"
         + "    signs=ranked[ranked.direction_abstentions==0].sort_values('direction_hit_rate',ascending=False)\n"
         + "    direction=signs.iloc[0] if len(signs) else None\n"
         + "    print(symbol, 'лидер MAE:',best['model'],'; доказательство против persistence:',best['verdict'])\n"
         + "    if direction is not None: print('Лидер направления:',direction['model'],f\"{direction['direction_population_correct']}/{direction['direction_population_n']}\",'; вывод:',direction['direction_verdict'])\n"
         + "print('Лидер по test — описание результата; модель serving по-прежнему выбрана до test на tune.')")
    md("""
### Графики каждого метода: обучение → test → inference

Левая колонка показывает реальные цены и OOF-прогнозы последнего expanding fold
**внутри train**: модель обучена до показанного validation-окна. Это честная
диагностика процесса обучения. Средняя — unseen test. Точки прогноза стоят
на timestamp **закрытия будущей свечи h15**, а не на origin; линии между ними
служат только визуальным ориентиром. Правая — последние 60 наблюдённых минут,
переходящие в полный h1…h15 прогноз с эмпирическим интервалом. Факт после cutoff
не подставляется в прогноз. Полные метрики считаются по всем origin, график
показывает предзаданные последние шесть часов, без выбора красивого участка.
""")
    code("plot_method_evidence(experiment,cv_traces,CFG)")
    md("""
## 5. Неопределённость и калибровка

**Ответ:** «Для каждого горизонта строю эмпирический 90% prediction interval по
абсолютным остаткам на отдельном calibration-периоде. Использую конечновыборочный
порядок ceil((n+1)×0.9), работаю в log-return space и перевожу границы в цену.
На untouched test смотрю coverage вместе с числителем/знаменателем, ширину
и interval score. Широкий интервал с высоким coverage сам по себе бесполезен.
Временная зависимость и drift не дают автоматически обещать conformal-гарантии;
поэтому интервалы называю эмпирическими и проверяю их качество forward».

Все модели здесь имеют одинаковую схему остаточной калибровки. Нативные
ARIMA/Prophet интервалы не смешиваются с ней под одним названием. Это не
полная predictive distribution; CRPS по симметричной полосе здесь не вычисляется.
""")
    code("interval_rows=[]\nfor r in experiment['results']:\n"
         + "    interval_rows.append({'symbol':r['symbol'],'model':r['model'],'covered_h15':r['PI90_covered_by_horizon'][-1],'n':r['PI90_denominator'],'coverage_h15':r['PI90_coverage_by_horizon'][-1],'interval_score_h15':r['PI90_interval_score_by_horizon'][-1],'width_h15_USDT':r['PI90_mean_width_USDT_by_horizon'][-1]})\n"
         + "display(pd.DataFrame(interval_rows))")
    md("""
## 6. Грязные данные, timestamps и drift

**Ответ:** «В исправлении этого проекта проверяю схему свечи, UTC-сетку, OHLC,
конечность значений, закрытие бара и полноту выгрузки. Идентичные дубли удаляю,
конфликтующие — отклоняю. Пропуски не закрываю будущими значениями и не сжимаю
временную ось: для этого benchmark запуск останавливается. При постоянном объёме
z-score становится нейтральным, вместо удаления целого origin. Реальный кейс
аудита: BTC/ETH/SOL в v2 имели сдвинутые на несколько минут периоды загрузки;
теперь все активы используют одну границу. Для live отдельно контролирую
freshness, ошибки API и качество прогнозов после созревания меток».

Экстремальные, но корректные рыночные движения не удаляются по квантилю всей
истории. Data/concept drift требует наблюдения distribution и forward residuals;
чувствительность detector сама должна быть настроена до test. Этот небольшой
сервис публикует ошибки, freshness, сдвиг признаков относительно train и coverage
после созревания будущих меток; полноценный внешний drift dashboard —
следующий операционный слой, а не уже выполненный эксперимент этого ноутбука.
""")
    code("# Контрфактическая проверка: произвольное изменение БУДУЩЕГО не меняет прошлые признаки.\n"
         + "for symbol in CFG.symbols:\n"
         + "    raw=market[symbol]; cutoff=len(raw)//2\n"
         + "    changed=raw.copy(); changed.loc[cutoff+1:,OHLCV]*=7\n"
         + "    a,b=prepare(raw,CFG),prepare(changed,CFG)\n"
         + "    pd.testing.assert_frame_equal(a.loc[:cutoff,FEATURES], b.loc[:cutoff,FEATURES])\n"
         + "    origins=a.iloc[[cutoff-1,cutoff]]\n"
         + "    np.testing.assert_array_equal(sequence_inputs(a,origins,CFG),sequence_inputs(b,origins,CFG))\n"
         + "print('PASS: будущие цены/объёмы не влияют на признаки и encoder прошлого')")
    md("""
## 7. Несработавшая исследовательская гипотеза

**Ответ:** «В этом benchmark гипотеза — что технические признаки или усложнение
модели дают преимущество перед persistence на горизонте 15 минут. Отказываюсь
от внедрения, если нет устойчивого OOS-прироста, даже при красивом графике или
лучшем train score. В исходном v2 статистические модели выдавали предупреждения
о несходимости; это технический FAIL, а не доказательство плохого forecasting
или успешный отрицательный эксперимент. В исправленной версии каждое состояние
явно отделено от результата исследования».

Дополнительный документированный отрицательный кейс из `claude_crypto_bot` —
поиск ранних признаков дневных «ракет» на историческом периоде 455 дней и 101
монете. Спецификация отмечает деградацию лучших пороговых правил после временного
разделения и не разрешает их внедрение. Это результат, зафиксированный авторами
репозитория; в этом ноутбуке тот большой backtest заново не выполняется.
[Источник](https://github.com/ostrowsky/claude_crypto_bot/blob/main/docs/specs/features/rocket-segment-spec.md).
""")
    code("# Автоматически формируем честную формулировку результата этого запуска.\n"
         + "for s in CFG.symbols:\n"
         + "    selected=experiment['choices'][s]\n"
         + "    found=[r for r in experiment['results'] if r['symbol']==s and r['model']==selected]\n"
         + "    if not found:\n        print(s, selected, 'FAIL: полного результата нет')\n        continue\n"
         + "    r=found[0]\n"
         + "    print(f\"{s}: tune выбрал {selected}; test MAE={r['MAE_h15_return']:.6g}, прирост={r['improvement_pct']:.2f}%, N={r['n_origins']}, дней={r['n_time_blocks']}, вывод={r['verdict']}\")")
    md("""
## 8. Production: сервис и мониторинг

**Ответ:** «Мой пример работающей архитектуры — проекты криптосигналов: Python,
асинхронный сбор Binance, Telegram, отдельные фоновые jobs, журналы событий,
replay и проверки качества. Это системы сигналов, а не подтверждённый опыт
управления капиталом. Для данного forecasting-демо подготовлен FastAPI-сервис:
замороженные модели, inference без обучения на test, закрытые свежие свечи,
API key, readiness и структурированные логи. Сервис проверен контрактными
тестами; непрерывная работа под реальной нагрузкой требует отдельного rollout».

Документированные исходные системы: [Claude bot](https://github.com/ostrowsky/claude_crypto_bot)
и [GPT bot](https://github.com/ostrowsky/gpt_crypto_bot).

**Runtime contract:** `/forecast/{symbol}` требует `X-API-Key`; неизвестный актив
→ 404, неверный ключ → 401, устаревшие данные/релиз → 503. Обновление всех активов
атомарно; пропуск дедлайна первого горизонта отклоняется. API различает доступность
бара и фактическую выдачу прогноза. Возраст входа ≤90 секунд, срок релиза ≤7 дней. Нет
автоматического переобучения в serving. Если выбранная на tune модель не имеет
положительной нижней границы diagnostic CI, API использует предзаданный
persistence fallback. Это критерий допуска прогноза, **не разрешение торговать**.

Процесс запускать с одним worker; TLS и rate limiting — на внешнем proxy.
Сырые online-снимки не копятся на диске. Внешний сбор логов, alerting, dashboard,
нагрузочная проверка и новые prospective данные — условия полноценной эксплуатации.
""")
    code("# Иллюстрация на ПОСЛЕДНЕМ ЗАКРЫТОМ баре фиксированного snapshot. Это не текущий live.\n"
         + "for symbol in CFG.symbols:\n    display(illustrative_forecast(experiment,symbol,CFG))")
    md("""
### Интерактивный inference: зафиксировать прогноз и дождаться факта

В активном Jupyter kernel выберите актив и методы и нажмите **«Зафиксировать
прогноз»**. Загружаются последние три дня закрытых свечей; все модели используют
исходные веса, будущие 15 точек сохраняются вместе с временем фактической выдачи
и уникальным ID в `forecast_demo_artifacts/prospective/`. Никакого переобучения.

Через минуту или позже нажмите **«Новые свечи → сравнить»**: новые фактические
цены появятся поверх исходных линий. Через 15 минут будут доступны все 15 точек.
Таблица показывает MAE/coverage только по созревшим точкам и их число. Прогноз
и его timestamp не изменяются при обновлении. Можно переключаться между
зафиксированными прогнозами и методами. Обновление ручное; Run All сам не
делает live-запросов и не ждёт 15 минут. После перезапуска kernel панель нужно
создать заново; сохранённые JSON остаются как журнал исходных прогнозов.

Это **prospective исследовательская проверка** старого релиза, не допуск к
торговле или обход срока действия production API. Фиксированный августовский
test не меняется. Виджеты требуют ipywidgets и активный Python kernel;
при открытии готового файла графики/таблицы видны, кнопки оживают после Run All.
""")
    code("live_comparison = interactive_comparison(experiment,CFG)")
    md("""
## 9. AI-инструменты последних 3–6 месяцев: Claude Code и Codex

**Ответ:** «Использовал Claude Code и Codex при разработке двух проектов:
claude_crypto_bot и gpt_crypto_bot. Применял их для исследования существующего
кода, оформления гипотез и спецификаций, подготовки Python-реализации,
тестов и проверки изменений. Сравнивал не убедительность ответа инструмента,
а результат: проходит ли код проверки, соблюдает ли временную причинность,
воспроизводится ли эксперимент. Вывод: AI ускоряет итерации, но его предложения
требуют независимой проверки данных и метрик; код и красивый отчёт сами по себе
не доказывают улучшения модели».

Основание — указанные пользователем проекты и подтверждение использования этих
инструментов. Контролируемого benchmark «Claude быстрее/лучше Codex на X%»
в изученных материалах нет; его не заявляем. Источники, за которыми кандидат
лично следит, из репозиториев установить нельзя — этот биографический пункт
нужно дополнить самостоятельно, если он понадобится рекрутеру.

[Claude project rules](https://github.com/ostrowsky/claude_crypto_bot/blob/main/CLAUDE.md) ·
[GPT project rules](https://github.com/ostrowsky/gpt_crypto_bot/blob/main/AGENTS.md).
""")
    md("""
## 10. Самая сильная автоматизация

**Ответ:** «Мой пример — автоматизация исследовательского контура в двух
криптопроектах: сбор свечей и событий, формирование датасетов, созревание будущих
меток, обучение моделей, replay, ежедневные отчёты и проверки достоверности.
Стек — Python, Binance API, CatBoost, Telegram, фоновые workers и планировщик;
Claude Code/Codex использовались как инструменты разработки. Результат —
воспроизводимый pipeline вместо ручного выполнения отдельных шагов, с явным
разделением исследовательской метрики и разрешения менять production-поведение.
Измеренное сокращение трудозатрат или доказанную доходность я здесь не заявляю».

В `claude_crypto_bot` описан EOD-конвейер snapshot → resolve → train → report.
В `gpt_crypto_bot` — ограниченный поиск CatBoost по validation Brier и независимые
проверки перед внедрением; training не получает право самостоятельно выпускать модель.
Состояние этих контуров неодинаково: наличие автоматических jobs не доказывает
полностью замкнутый, успешно улучшающийся production-loop.

[Claude daily pipeline](https://github.com/ostrowsky/claude_crypto_bot/blob/main/docs/specs/features/daily-learning-pipeline-spec.md) ·
[GPT optimization spec](https://github.com/ostrowsky/gpt_crypto_bot/blob/main/docs/specs/prediction-error-optimization-loop.md).

**Результат текущего аудита GPT bot:** полный Truth Harness — **FAIL TH-11**
из-за несовпадения source hash портфельного replay. Это не исправляется данным
ноутбуком и исключает использование старой portfolio alpha как достижения.
Полный runtime-аудит Claude bot в этом задании не выполнялся; его документы
использованы как источники об устройстве, не как подтверждение текущих метрик.
""")
    md("""
## Запуск API отдельно от эксперимента

После Run All создайте отдельный секрет `FORECAST_API_TOKEN` длиной ≥24 символов
в окружении процесса; не сохраняйте его в ноутбук. Передайте `experiment` в
`create_app(experiment, CFG)` и запускайте Uvicorn одним worker. В Jupyter удобнее
экспортировать встроенный модуль и запустить его отдельным процессом: это избегает
конфликтов event loop и сохраняет воспроизводимый startup.

При использовании репозитория:
```text
python -m pip install -r research/requirements-forecast.txt
python files/research_forecast.py --serve --end-utc <заранее-зафиксированный-свежий-UTC-cutoff>
```
Docker recipe: `research/Dockerfile.forecast` (build context — корень репозитория).
Образ по этому recipe в текущей среде не собирался: Docker не доступен.
Обычный inference/API и основные экспериментальные модели проверяются отдельно.

Для новой даты релиза заранее зафиксируйте cutoff и протокол и повторите
исследование. Повторное подглядывание в один test превращает его в validation;
для следующего подтверждения используйте новый sealed forward-период.
""")
    code("# Экспорт выполняется только по явному включению. Секреты в файлы не попадают.\n"
         + "EXPORT_SERVICE = False\n"
         + "if EXPORT_SERVICE:\n"
         + "    notebook_path=next(iter(Path.cwd().glob('Binance_BTC_ETH_SOL_ML_Researcher_Demo*.ipynb')),None)\n"
         + "    if notebook_path is None: raise FileNotFoundError('Поместите notebook в текущую рабочую папку kernel')\n"
         + "    nb=json.loads(notebook_path.read_text(encoding='utf-8'))\n"
         + "    engine=next(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code' and ''.join(c['source']).startswith('\\\"\\\"\\\"Isolated, causal'))\n"
         + "    Path('research_forecast.py').write_text(engine+'\\nif __name__ == \\\"__main__\\\":\\n    main()\\n',encoding='utf-8')\n"
         + "    print('Запустите отдельным процессом: python research_forecast.py --serve')")
    md("""
## Критерии завершённого demo-run

1. Полные input snapshots всех трёх активов, общий UTC период, hashes.
2. Все split-границы проходят label-availability assertion.
3. Для каждого завершившегося model/symbol одинаковые test-origin.
4. В таблице есть persistence, интервалы, sample counts и честный verdict.
5. Модели/версии/ошибки сохраняются в отдельный JSON; веса и данные не коммитятся.
6. Будущие возмущения не меняют прошлые признаки.
7. API не выдаёт устаревший прогноз; UNKNOWN сохраняет безопасный baseline.

В основном benchmark достаточно временных блоков для расчёта CI. Если CI
пересекает ноль, итог `INCONCLUSIVE`: превосходство не подтверждено. Это
отличается от прежнего `UNKNOWN` из-за четырёх дней данных. Нельзя менять
историю, метрику или seed после результата, пока не получится значимость.
""")
    for i,cell in enumerate(cells):
        cell['id'] = f"demo-{i:03d}"
    return dict(nbformat=4, nbformat_minor=5, cells=cells,
                metadata=dict(kernelspec=dict(display_name="Python 3.11",language="python",name="python3"),
                              language_info=dict(name="python",version="3.11")))


def build(release=None):
    """Default current inference; delivered artifact supplies the frozen release."""
    old=build_research()
    cells=[]
    def md(text):cells.append(dict(cell_type='markdown',metadata={},source=text.strip().splitlines(keepends=True)))
    def code(text,hidden=False):cells.append(dict(cell_type='code',metadata={'jupyter':{'source_hidden':True}} if hidden else {},execution_count=None,outputs=[],source=text.strip().splitlines(keepends=True)))
    md('''# ML Researcher · BTC / ETH / SOL · версия 3.2

**Текущий прогноз h1…h15 из одного момента, одинаковое UTC-окно у всех методов.**
ARIMA, ETS (Gaussian state-space), SARIMA, SARIMAX, Prophet, Ridge, XGBoost,
LSTM и настоящий Temporal Fusion Transformer (PyTorch Forecasting).
Вероятностная часть: отдельные эмпирические 90% prediction intervals на каждом горизонте.
Persistence удалена из прогнозов, графиков и выбора методов; ошибка последней
известной цены остаётся внутренним контролем пользы модели.

**Запуск:** Python 3.11, зависимости из следующей ячейки → Restart Kernel → Run All.
В готовом файле обученные веса встроены. Повторный запуск загружает веса и получает
текущие закрытые свечи; обучение не тратит будущие 15 минут. Все активы используют
одну исходную минуту, фактическое время выдачи записывается отдельно. Горизонты —
следующие 15 закрытий минутных свечей, а не переименованные timestamps с секундами.
Через 1–15 минут кнопка «Новые свечи → сравнить» добавляет факт к исходному прогнозу.
Будущего факта до закрытия свечей нет: метрики помечены PENDING/PARTIAL и имеют знаменатели.

**Исправление вводящей в заблуждение картинки:** исторические h15-точки из разных
моментов больше не соединяются в «траекторию». Даже нулевой прогноз доходности
повторял бы цену при таком построении. Теперь исторический пример — одна полная
15-минутная траектория, как в live. Близкий к константе результат модели сохраняется
как сырой выход и отмечается NO_INFORMATIVE_SIGNAL; случайные изгибы не добавляются.
Кандидат с заметным движением также не считается доказанно полезным.
''')
    requirements=(ROOT/'research/requirements-forecast.txt').read_text(encoding='utf-8')
    packages=[x for x in requirements.splitlines() if x and not x.startswith('#')]
    code('# Выполнить один раз в отдельном kernel, затем перезапустить:\n# %pip install '+' '.join(packages)+"\nimport importlib.util\nrequired=['numpy','pandas','scipy','sklearn','xgboost','statsmodels','matplotlib','ipywidgets','prophet','torch','lightning','pytorch_forecasting','safetensors']\nmissing=[p for p in required if importlib.util.find_spec(p) is None]\nif missing: raise RuntimeError('Установите зависимости: '+', '.join(missing))")
    source=(ROOT/'files/research_forecast.py').read_text(encoding='utf-8').rsplit('\nif __name__ == "__main__":',1)[0]
    source+='\n\n'+(ROOT/'files/research_forecast_release.py').read_text(encoding='utf-8')
    preparation=(ROOT/'files/prepare_research_forecast_release.py').read_text(encoding='utf-8')
    body=preparation.split('    args=parser.parse_args()\n',1)[1].rsplit('\n\nif __name__',1)[0]
    aliases=sorted(set(re.findall(r'\brf\.([A-Za-z_]\w*)',body)))
    source+='\n\ndef rebuild_demo_release(output_dir="forecast_demo_artifacts", historical_end="2026-09-01T00:00:00Z", release_end=None):\n'
    source+='    import types\n    rf=types.SimpleNamespace(**{name:globals()[name] for name in '+repr(aliases)+'})\n'
    source+='    args=types.SimpleNamespace(output=output_dir,historical_end=historical_end,release_end=release_end or pd.Timestamp.now(tz="UTC").floor("D").isoformat())\n'
    source+=body+'\n    return envelope\n'
    code(source,hidden=True)
    code("SOURCE_SHA256 = "+repr(hashlib.sha256(source.encode()).hexdigest())+'\nEMBEDDED_RELEASE = '+repr(release)+'''\nRELEASE_PATH=Path('forecast_demo_artifacts/release_v32.json')
REBUILD_RELEASE=False  # отдельное, длительное обучение; текущий forecast создаётся ПОСЛЕ него
if REBUILD_RELEASE:
    EMBEDDED_RELEASE=rebuild_demo_release()
if EMBEDDED_RELEASE is None:
    if not RELEASE_PATH.exists():
        raise FileNotFoundError('Это чистая исходная версия. Включите REBUILD_RELEASE=True для подготовки весов либо откройте выполненный v3_2 notebook со встроенными весами.')
    EMBEDDED_RELEASE=json.loads(RELEASE_PATH.read_text(encoding='utf-8'))
experiment,CFG,release_provenance=unpack_release(EMBEDDED_RELEASE)
from IPython.display import display
display(pd.DataFrame([dict(model=n,kind={'ETS':'state-space ETS(A,Ad,N)','SARIMAX':'calendar state-space recurrence','XGBoost':'boosting','LSTM':'DL','TFT':'DL + attention'}.get(n,n)) for n in CFG.models]))
print('Обучение:',release_provenance['train_start'],'→',release_provenance['train_cutoff'])
print('Tune до:',release_provenance['tune_cutoff'],'; calibration до:',release_provenance['calibration_cutoff'])
print('Кандидат для prospective проверки; старый benchmark не подтверждает новые веса.')
''',hidden=True)
    md('''## Общий текущий прогноз — все девять методов

Сначала фиксируется один origin для всех активов. Если расчёт/загрузка пропустили
первое закрытие, batch отклоняется целиком: его нельзя задним числом назвать прогнозом.
В таком случае повторите эту ячейку для явно нового общего момента.
Цены показаны в USDT и в процентах от origin, без подмены модельного выхода.
Диагностический порог слабого сигнала задан заранее: максимальное движение <0.01%
или отношение движения к h15 полуширине интервала <0.1. Это критерий отображения,
не статистический тест преимущества и не торговый сигнал.
''')
    code("live_comparison=ForecastComparison(experiment,CFG,output_dir='forecast_demo_artifacts/prospective')\nCURRENT_BATCH=live_comparison.create_batch()\nprint(json.dumps(live_comparison.batches[CURRENT_BATCH],ensure_ascii=False,indent=2))\ndisplay(live_comparison.batch_table(CURRENT_BATCH))\nlive_comparison.plot_batch(CURRENT_BATCH)")
    code("horizon_rows=[]\nfor key in live_comparison.batches[CURRENT_BATCH]['snapshot_ids']:\n    snap=live_comparison.snapshots[key]\n    for name,path in snap['paths'].items():\n        for h,target in enumerate(snap['target_close_at']):\n            horizon_rows.append(dict(symbol=snap['symbol'],model=name,horizon=h+1,target_close_at=target,predicted_close=path['price'][h],PI90_lower=path['lower'][h],PI90_upper=path['upper'][h]))\ndisplay(pd.DataFrame(horizon_rows))\nfor key in live_comparison.batches[CURRENT_BATCH]['snapshot_ids']:\n    snap=live_comparison.snapshots[key];path=snap['paths']['TFT']\n    print(snap['symbol'],'TFT: native one-minute log-return quantiles, % (not cumulative price quantiles)')\n    display(pd.DataFrame(100*np.asarray(path['native_minute_return_quantiles']),index=snap['target_close_at'],columns=[str(q) for q in path['native_quantile_levels']]))")
    code("interactive_comparison(experiment,CFG,controller=live_comparison)")
    md('''## Проверка отсутствия подглядывания и воспроизводимости

При подготовке каждого из 27 model/asset сочетаний все будущие OHLCV после origin
заменены и повторно рассчитаны признаки. Прогноз сравнивается также с расчётом
только на закрытом префиксе. Оба результата должны совпасть. Отдельно проверяется
совпадение прогнозов до/после переноса весов. Ни test, ни факт после выдачи не меняют
scaler, веса или параметры. TFT decoder получает будущий календарь и замороженные
значения origin; будущие рыночные данные не передаются даже как «заглушка».
Веса хранятся как численные состояния, native XGBoost/Prophet JSON и safetensors.
''')
    code("display(pd.DataFrame(release_provenance['causality']))\ndisplay(pd.DataFrame(release_provenance['roundtrip']))\nassert len(release_provenance['causality'])==len(CFG.models)*len(CFG.symbols)\nassert all(r['future_mutation']=='PASS' and r['closed_prefix']=='PASS' for r in release_provenance['causality'])")
    md('''## Историческая сравнительная таблица и качество моделей

Эта таблица относится к августу 2026, **не к текущим новым весам**.
120 дней: 60 train / 15 tune / 15 calibration / 30 полных test-дней.
Добавленные модели проверяются на уже просмотренном периоде; их результат —
ретроспективный benchmark, не новый независимый sealed holdout.
Новые веса подготовлены отдельно на 42 днях: 35 train / 4 tune / 3 calibration.
Для live требуется лишь 3 дня контекста. Сроки выбраны для этого демо, не доказаны
как универсально минимальные. Данные и параметры не добавляются до получения
желаемой значимости. Дальнейшее подтверждение — только новый forward период.

Сравниваются ошибка h15 log-return, USDT MAE, RMSE, direction correct/N,
абстенция, coverage и interval score. Самая малая MAE — описательный победитель;
доказательство устойчивого преимущества требует положительного corrected CI.
''')
    code("benchmark=pd.DataFrame([r for r in experiment['results'] if r['model'] in CFG.models])\nbenchmark['RMSE_h15_return']=benchmark.RMSE_by_horizon.map(lambda x:x[-1])\nbenchmark['PI90_covered_h15']=benchmark.PI90_covered_by_horizon.map(lambda x:x[-1])\nbenchmark['PI90_coverage_h15']=benchmark.PI90_coverage_by_horizon.map(lambda x:x[-1])\nbenchmark['PI90_interval_score_h15']=benchmark.PI90_interval_score_by_horizon.map(lambda x:x[-1])\ncolumns=['symbol','model','n_origins','MAE_h15_return','MAE_h15_USDT','baseline_MAE_h15_return','RMSE_h15_return','improvement_pct','n_time_blocks','verdict']\ndisplay(benchmark[columns].sort_values(['symbol','MAE_h15_return']))\nplot_comparison(experiment,CFG)\nfor symbol in CFG.symbols:\n    group=benchmark.loc[benchmark.symbol==symbol]\n    winner=group.sort_values('MAE_h15_return').iloc[0]\n    direction=group.sort_values('direction_hit_rate',ascending=False).iloc[0]\n    print(symbol,'минимальная MAE:',winner.model,'; verdict:',winner.verdict)\n    print('Направление:',direction.model,str(direction.direction_population_correct)+'/'+str(direction.direction_population_n),'вывод:',direction.direction_verdict)\nprint('Ошибка без модели — только внутренний контроль; Persistence отсутствует в списке прогнозов.')")
    code("details=['symbol','model','direction_population_correct','direction_population_n','direction_abstentions','direction_balanced_accuracy','direction_ci95_familywise_gain_pp','direction_verdict','n_calibration','PI90_covered_h15','PI90_denominator','PI90_coverage_h15','PI90_interval_score_h15','native_quantile_pinball','native_quantile_crossings','native_quantile_comparisons']\ndisplay(benchmark[[c for c in details if c in benchmark]])\ndisplay(pd.DataFrame(experiment['direction_baselines']))\ndisplay(pd.DataFrame([r for r in release_provenance['cv_results'] if r['model'] in CFG.models]))")
    md('''### Исторические price-графики: один origin вместо соединённых h15

Отдельная архивная диагностика (по умолчанию скрыта, чтобы не смешивать её с общим
текущим окном). Включите SHOW_HISTORICAL=True: TRAIN OOF и TEST используют все
15 горизонтов одного origin, одинакового у всех методов внутри каждого этапа.
Исходная точка выбрана по времени до просмотра ошибок, а не по удачной картинке.
''')
    code('''SHOW_HISTORICAL=False
if SHOW_HISTORICAL:
    import matplotlib.pyplot as plt
    for symbol in CFG.symbols:
        fig,axes=plt.subplots(len(CFG.models),2,figsize=(14,3*len(CFG.models)),squeeze=False)
        for i,name in enumerate(CFG.models):
            for col,stage in enumerate(('train_oof','test')):
                ex=release_provenance['examples'][symbol][name][stage]
                origin=utc(ex['origin']);end=origin+pd.Timedelta(minutes=15)
                ax=axes[i,col]
                ax.plot(pd.DatetimeIndex(ex['actual_times']),ex['actual'],color='black',label='Actual')
                ax.plot(pd.date_range(origin,periods=16,freq='min'),[ex['price']]+ex['prediction'],'o--',ms=2,label='One origin: h1..h15')
                ax.axvline(origin,color='gray',ls=':');ax.set_xlim(origin-pd.Timedelta(minutes=15),end)
                ax.set_title(name+' | '+stage+' | '+origin.isoformat(),fontsize=9);ax.legend();ax.grid(alpha=.2)
        fig.suptitle(symbol+' | archived single-origin forecasts');fig.tight_layout();plt.show()
''')
    md('''## Ответы на 10 вопросов

Примеры ниже относятся к воспроизводимому демо и документированным проектам.
Числа берутся из таблицы выше. Наличие кода не означает подтверждённый прирост,
а статистическая метрика не означает доходность торгового портфеля.
''')
    # Preserve interview answers and provenance, updated for the actual model set.
    for cell in old['cells']:
        text=''.join(cell['source'])
        if cell['cell_type']=='markdown' and any(text.startswith(f'## {i}.') for i in range(1,11)):
            text=text.replace('Prophet, LSTM и TFT\nподготовлены как опциональные эксперименты.', 'Prophet, LSTM и TFT включены в основной сравнительный эксперимент, добавлен ETS(A,Ad,N).')
            text=text.replace('DL — отдельные явно включаемые гипотезы.', 'ETS, Prophet, LSTM и TFT включены в сравнение.')
            text=text.replace('SARIMAX использует только известный\nкалендарь: реальные будущие объёмы и цены в exog не передаются.',
                              'SARIMAX получает будущий календарь и замороженные рыночные признаки origin; реальные будущие объёмы и цены в exog не передаются.')
            text=text.replace('Финальный период — август 2026;', 'Период ретроспективного benchmark — август 2026;')
            text=text.replace('30 untouched test.', '30 test-дней (период уже просмотрен; новый независимый holdout нужен отдельно).')
            text=text.replace('Финальный test не участвует в поиске гиперпараметров.', 'Августовский test не используется для настройки; для добавленных моделей это ретроспективное сравнение.')
            text=text.replace('по новой таблице.', 'по новой исторической таблице; текущие новые веса оцениваются отдельно.')
            text=text.replace('benchmark включает persistence,\n', 'benchmark включает\n')
            text=text.replace('ETS или отдельную state-space разработку как свой\nподтверждённый опыт по этому ноутбуку не заявляю',
                              'ETS(A,Ad,N) реализован через Gaussian state-space в statsmodels; параметры обучаются только на train, затем фильтр обновляет состояние по наблюдаемой истории')
            text=text.replace('Опциональные DL-адаптеры не считаются проверенными только потому, что код написан.',
                              'LSTM и TFT реально обучены и рассчитаны; для обоих проверены будущие возмущения и перенос весов.')
            text=text.replace('На untouched test', 'На отложенном ретроспективном test')
            text=text.replace('перед persistence', 'перед контрольной ошибкой последней цены')
            text=text.replace('persistence fallback', 'fallback последней известной цены')
            text=text.replace('нет TFT/LSTM/Prophet-результатов', 'результаты TFT/LSTM/Prophet/ETS показаны в общей таблице')
            text=text.replace('финальный test не участвует', 'ретроспективный test не участвует')
            if text.startswith('## 2.'):
                text+='\n\nProphet: фиксированная train-only модель, прогнозирует изменение своего log-price тренда/сезонности относительно origin; уровень привязан только к наблюдённой цене origin. ETS: параметры Gaussian state-space MLE на train, causal Holt filter на доступном контексте. TFT: train-only scalers, fixed 15-minute grid и последние 7 train-дней для ограничения CPU, checkpoint по tune.\n'
            if text.startswith('## 7.'):
                text+='\n\nСтарые h15-картинки были исправлены: сходство ряда точечных прогнозов с уровнем цены не подтверждает прогнозирование будущего движения. Нулевая доходность тоже создаёт такое сходство.\n'
            if text.startswith('## 5.'):
                text+='\n\nTFT дополнительно прогнозирует семь нативных квантилей минутной доходности, обучаясь с QuantileLoss. Оцениваю pinball loss и crossing count/число пар. Квантили минутных доходностей не складываю в якобы квантили цены; интервалы накопленной траектории калибрую отдельно.\n'
            md(text)
    md('''## Повторение обучения и отдельный production API

В репозитории `python files/prepare_research_forecast_release.py` пересоздаёт
исторический benchmark, train OOF, свежий research release и проверки причинности.
Подготовка весов — отдельный процесс, Run All готового файла не переобучает модели.
Этот же pipeline встроен в ноутбук: явное REBUILD_RELEASE=True запускает подготовку
с нуля без репозитория. Это длительный исследовательский режим; текущий 15-минутный
forecast начинается после подготовки. Обычный запуск использует готовые веса.
Production архитектура FastAPI сохранена во встроенном коде: readiness, auth,
freshness, дедлайн первого горизонта, expiry, атомарное обновление и mature-label
monitoring. Неизвестное преимущество сохраняет внутренний безопасный fallback;
удаление Persistence из исследовательской панели не ослабляет serving gates.
Для полноценного допуска этих девяти методов нужен новый независимый forward test.
Docker/нагрузочный rollout в этой работе не подтверждены.
''')
    # Preserve the standalone module export contract (second-to-last cell).
    export=next(''.join(c['source']) for c in old['cells'] if c['cell_type']=='code' and 'EXPORT_SERVICE = False' in ''.join(c['source']))
    code(export)
    md('''Полный аудит бота: **FAIL TH-11** (существующий portfolio replay hash).
Это независимая от демо проблема; она не превращается в PASS благодаря новым графикам.
Вывод о forecasting определяется численными ошибками и forward-проверкой, а не формой линии.
''')
    for i,c in enumerate(cells):c['id']=f'demo-v32-{i:03d}'
    return dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=old['metadata'])


def main():
    TARGET.parent.mkdir(parents=True,exist_ok=True)
    TARGET.write_text(json.dumps(build(),ensure_ascii=False,indent=1)+"\n",encoding="utf-8")
    print(TARGET)


if __name__ == "__main__":
    main()
