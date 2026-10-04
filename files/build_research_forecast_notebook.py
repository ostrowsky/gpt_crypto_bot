"""Build a standalone, clean-output interview demo from reviewed research code."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "research" / "Binance_BTC_ETH_SOL_ML_Researcher_Demo_v3.ipynb"


def build():
    cells = []
    def md(text):
        cells.append(dict(cell_type="markdown", metadata={}, source=text.strip().splitlines(keepends=True)))
    def code(text, hidden=False):
        cells.append(dict(cell_type="code", execution_count=None, outputs=[],
                          metadata={"jupyter": {"source_hidden": True}} if hidden else {},
                          source=text.strip().splitlines(keepends=True)))
    md("""
# ML Researcher: от сырых свечей до проверяемого прогноза и API

**Демо для интервью · BTCUSDT / ETHUSDT / SOLUSDT · версия 3**

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
         + "required = ['numpy', 'pandas', 'scipy', 'sklearn', 'xgboost', 'statsmodels', 'matplotlib']\n"
         + "missing = [p for p in required if importlib.util.find_spec(p) is None]\n"
         + "if missing:\n    raise RuntimeError('Установите зависимости из команды выше: ' + ', '.join(missing))")
    source = (ROOT / "files" / "research_forecast.py").read_text(encoding="utf-8")
    source = source.rsplit('\nif __name__ == "__main__":', 1)[0]
    code(source, hidden=True)
    code(f"SOURCE_SHA256 = {hashlib.sha256((ROOT/'files'/'research_forecast.py').read_text(encoding='utf-8').encode('utf-8')).hexdigest()!r}\n"
         + "CFG = ForecastConfig()\n"
         + "# End exclusive: [2026-09-04 00:00 UTC, 2026-10-04 00:00 UTC).\n"
         + "# Для НОВОГО эксперимента измените дату ДО просмотра результатов.\n"
         + "# CFG = ForecastConfig(history_days=120, end_utc='2026-10-04T00:00:00Z')\n"
         + "# CFG = ForecastConfig(models=('Persistence','Ridge','XGBoost','ARIMA','SARIMA','SARIMAX','LSTM','TFT'), classical_window=2880)\n"
         + "CACHE = Path('forecast_demo_artifacts/cache')\n"
         + "EVIDENCE = Path('forecast_demo_artifacts/benchmark.json')\n"
         + "print(json.dumps(asdict(CFG), ensure_ascii=False, indent=2))")
    md("""
## 1. Сильный проект по forecasting — задача, данные, baseline, результат

**Ответ:** «Мой пример — исследование краткосрочного прогноза BTC, ETH и SOL:
минутные OHLCV Binance, горизонт 1–15 минут, цель — накопленная лог-доходность
от последнего закрытого бара. Baseline — неизменность цены. Сравниваю Ridge,
XGBoost и ARIMA, а сезонные и DL-модели — как отдельные явно включаемые гипотезы.
Результат оцениваю по ошибке на одинаковых будущих timestamp, а не по похожести
графиков. Числа привожу из таблицы ниже; стабильное превосходство заранее не обещаю».

Тридцать дней — ограниченный демонстрационный период, не вся история биржи.
Для утверждения об устойчивости нужен новый заранее зафиксированный длинный
период; пример 120 дней указан выше. В этом запуске торговые правила не меняются.
""")
    code("market, input_manifest = {}, {}\nfor symbol in CFG.symbols:\n"
         + "    market[symbol], input_manifest[symbol] = fetch_history(symbol, CFG, CACHE)\n"
         + "    print(symbol, len(market[symbol]), input_manifest[symbol]['sha256'])\n"
         + "display(pd.DataFrame([{ 'symbol':s, 'rows':len(d), 'start':d.open_time.iloc[0], 'end':d.open_time.iloc[-1], 'missing_minutes':0 } for s,d in market.items()]))")
    md("""
## 2. Честный backtest и доступность данных

**Ответ:** «Разбиваю данные по времени: 60% train, 15% tune, 10% calibration,
15% untouched test. Train обучает модель и scaler; tune выбирает модель/эпоху;
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

Фиксированные ML-модели и rolling refit классических моделей — разные
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
Для доверительного вывода задан минимум **10 полных test-дней**; даже это
минимальный диагностический порог, не гарантия независимости дней.
Default 30 дней дают меньше — корректный вывод будет `UNKNOWN`.
""")
    code("experiment = run_experiment(market, CFG)\n"
         + "save_evidence(experiment, input_manifest, EVIDENCE)\n"
         + "benchmark = pd.DataFrame(experiment['results'])\n"
         + "display(benchmark[['symbol','model','n_origins','MAE_h15_return','MAE_h15_USDT','improvement_pct','selected_before_test','n_time_blocks','verdict']])\n"
         + "display(pd.DataFrame(experiment['status']))\n"
         + "print('Параметры/версии/input hashes:', EVIDENCE.resolve())")
    code("cv_results = expanding_window_check(experiment, CFG, model_names=('Ridge',))\n"
         + "# Добавляйте XGBoost в fold-протокол ДО нового эксперимента, если проверяете его устойчивость.\n"
         + "display(pd.DataFrame(cv_results)[['symbol','model','fold','n_train','n_origins','improvement_pct']])")
    md("""
## 4. Простые и сложные подходы

**Ответ:** «В этом проекте основной воспроизводимый benchmark включает persistence,
линейную Ridge, boosting XGBoost и ARIMA. SARIMA/SARIMAX, Prophet, LSTM и TFT
подготовлены как опциональные эксперименты. SARIMAX использует известный календарь
и состояние на origin, замороженное на горизонт. Интервал 60 минут у SARIMA —
гипотеза, не установленный факт сезонности. Сложность принимаю только после
устойчивого OOS-прироста. ETS или отдельную state-space разработку как свой
подтверждённый опыт по этому ноутбуку не заявляю».

Опциональные DL-адаптеры не считаются проверенными только потому, что код написан.
Результат `UNAVAILABLE` или `FAIL` виден явно. TFT здесь обучается отдельно для
каждого актива: это исключает случайное несоответствие временных границ panel.
После каждого decoder окна проверяется точный timestamp. Prophet с суточной
сезонностью требует минимум 2880 минут истории на fit; 12 часов недостаточно.
""")
    code("display(comparable_ranking(experiment['results'], CFG))\n"
         + "import matplotlib.pyplot as plt\n"
         + "pivot=benchmark.pivot(index='model',columns='symbol',values='improvement_pct')\n"
         + "ax=pivot.plot.bar(figsize=(11,4))\nax.axhline(0,color='black',linewidth=1)\n"
         + "ax.set_ylabel('Снижение MAE log return, %')\n"
         + "ax.set_title('Одинаковые test-origin; положительное число ещё не доказывает устойчивость')\nplt.tight_layout(); plt.show()")
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
python files/research_forecast.py --serve --end-utc 2026-10-04T00:00:00Z
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
         + "    notebook_path=Path('Binance_BTC_ETH_SOL_ML_Researcher_Demo_v3.ipynb')\n"
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

`UNKNOWN` — полноценный исследовательский результат. Он означает, что данных
недостаточно для заявленного вывода, и не заменяется оптимистичным формулированием.
""")
    for i,cell in enumerate(cells):
        cell['id'] = f"demo-{i:03d}"
    return dict(nbformat=4, nbformat_minor=5, cells=cells,
                metadata=dict(kernelspec=dict(display_name="Python 3.11",language="python",name="python3"),
                              language_info=dict(name="python",version="3.11")))


def main():
    TARGET.parent.mkdir(parents=True,exist_ok=True)
    TARGET.write_text(json.dumps(build(),ensure_ascii=False,indent=1)+"\n",encoding="utf-8")
    print(TARGET)


if __name__ == "__main__":
    main()
