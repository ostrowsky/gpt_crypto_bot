# Binance Demo: самостоятельное приложение и границы изоляции

Дата: 2026-10-10. Status: **P0-A KERNEL_IMPLEMENTED; full integration isolation pending**.
Owner: repository maintainer. Application ID: `binance_demo_bot`.
Приоритет: обязательное условие фаз 0–7
[программы](binance-demo-autonomous-trading-program.md).
Основание: пользователь требует отдельное приложение без воздействия
на текущую версию бота. Создан собственный package/env/path/credential kernel;
полная process/account/resource изоляция ещё не подтверждена.

## Problem and objective fit

Встраивание demo adapter в `files/monitor.py`, импорт `files/config.py`, общий
`.env`, ledger, Telegram poller или learning worker связывают новую торговлю
с действующим ботом. Его рестарт, состояние и качество не должны зависеть
от разработки нового financial learner. Изоляция — prerequisite дневного PnL
и доказательной атрибуции, а не новая торговая гипотеза.

## Scope and application layout

Самостоятельный Python project внутри репозитория: `apps/binance_demo_bot`.
Общий Git-хостинг не означает общие процессы, данные или зависимости.
Планируемая структура:

```text
apps/binance_demo_bot/
  pyproject.toml, dependency lock, .gitignore, README.md
  src/binance_demo_bot/     # собственные adapter/data/policy/OMS/ledger/learning
  tests/                   # свои fixtures и implementation tests
  config/                  # несекретные versioned contracts/pilot profiles
  .env.example             # только имена/пустые значения
  .env                     # local credentials, ignored
  .venv/                   # отдельная среда Python, ignored
  .runtime/                # db, outbox, cursors, locks, PID, snapshots, models,
                           # reports, logs, backups, cache и temporary files
```

У приложения свои executable entrypoint, CWD, dependency lock, bootstrap,
supervisor/service names, health endpoint, UI port, stop markers и version.
Оно запускается и тестируется без запуска или импорта старого приложения.
`files/`, старый `pyembed`, `.venv`, root `.runtime`, positions/models/logs
и штатные launch/learning scripts не являются его runtime dependencies.

Старые алгоритмы/спецификации разрешены как документированные источники идей.
Портирование — самостоятельный код/fixtures в новом package с новым source
hash и проверками; никаких live imports, sys.path hacks, symlinks к старому
state или автоматического наследования изменений текущего бота.
Поставка этих specs не мигрирует старые workers или данные и не запускает
новый сервис. Current application source/config/startup остаются прежними.

## Configuration, state and credential contract

Явный `APP_ROOT` выводится из установленного package/config, не из случайного
текущего каталога. Все writable/read-runtime paths после resolve остаются
в собственном root; traversal и symlink escape запрещены. Независимо
проверяются environment/host allowlist и secret redaction фаз 1/3.

Загрузка только из `apps/binance_demo_bot/.env` либо явно переданной пары
переменных процесса нового приложения. Запрещены `find_dotenv`/поиск выше root,
`files/.env`, inherited legacy config/PYTHONPATH и fallback к production key.
Не менять global Python/pip/user environment, текущие scheduled tasks и сервисы.
Новый `.gitignore` обязан исключать secrets, local environment и всё runtime
до создания `.env`/state; runtime и model artifacts не коммитятся.

Ранее ключи были сохранены в старом `files/.env`. Новый runtime этот файл
не читает. Возможный explicit provisioning копирует только demo-пару в новый
ignored файл без вывода значений; старый `.env` не удаляется и не редактируется.
Копирование сейчас не выполнялось. Credential generation/capability evidence
принадлежат новому application ID; old ticket/model/ledger не принимаются.

Каждый EventEnvelope включает `application_id=binance_demo_bot`, episode/scope,
process instance и остальные поля программы. Source/artifact hashes охватывают
собственный package, lock и frozen contracts, не весь изменяющийся старый repo.
Отчёт или другой git commit старой версии не меняют новую policy generation.

## Process, account and resource isolation

One OMS owner и takeover относятся исключительно к экземплярам нового app.
Bootstrap/recovery/stop не могут attach/kill/restart старые bot/collector/
learning processes. PID проверяется по executable/CWD/application instance
identity; просто совпавший PID или `python.exe` не является своим worker.
Unknown owner вызывает собственный startup failure, не зачистку чужих locks.

Trading account должен быть выделен новому app и иметь exclusive ownership.
Другой API key не создаёт другой счёт. Нельзя присвоить себе чужие orders/lots
или отменить их для выполнения своего risk cap. Unknown/manual/other-app
activity вызывает reconcile/attribution incident и block new entries/promotion.
Seed assets учитываются по P0; управление ими требует scope ownership manifest.
Никакое выключение приложения не удаляет принадлежащие ему exchange stops.

UI — отдельный локальный bind/порт с собственным auth и health. При занятом
порту startup FAIL, чужой listener не завершается. Если нужен Telegram,
используется отдельный bot token/poller; старый токен/меню/getUpdates не
подключаются. Server keys и notifications не наследуются из прежнего `.env`.

Настраиваются конечные CPU/memory/disk/worker/training/rate budgets до запуска.
IP/connection и account limits могут быть общими несмотря на разные ключи;
обрабатывать aggregate exchange headers/Retry-After, оставлять headroom для
защиты и ограничивать собственные backfills/training. Исчерпание бюджета
останавливает собственную работу/входы, не старые collectors. Отдельный процесс
не гарантирует отсутствие нагрузки на общий компьютер/сеть: это измеряется
совместным испытанием, а не объявляется по структуре каталогов.

## Primary metrics and acceptance criteria

### P0-A isolation kernel: первый исполняемый scope

Контракт фиксируется до тестов/кода. Собственный runtime использует Python
3.14 из отдельного `.venv`, editable package и versioned `uv.lock`; старый
`pyembed` допустим только как внешний engineering runner. Сборщик фиксируется
в `pyproject.toml`; cache находится в собственном `.runtime`. Автоматического
запуска сервисов/торговли, provisioning реальных ключей и shared poller нет.

Публичный модуль `binance_demo_bot.isolation`:

- `AppPaths(root)`: explicit абсолютный root приложения, canonical resolve,
  без зависимости от CWD. `runtime_path(relative)` возвращает путь только
  внутри собственного `.runtime`; absolute paths, `..` и resolved escapes
  (включая links/junctions) отвергаются `ValueError`. `env_file` возвращает
  только собственный `.env`, с такой же проверкой resolved boundary.
- `assert_clean_pythonpath(value)`: null/пустое значение допустимо;
  непустой injected PYTHONPATH отвергается `ValueError`. Проверяется на entrypoint
  до импорта runtime компонентов; директория tests не является production startup.
- `load_demo_credentials(paths, explicit=None)`: только собственный `.env`,
  либо явно переданный mapping с полной парой `BINANCE_DEMO_API_KEY` и
  `BINANCE_DEMO_API_SECRET`. Нет поиска parent, legacy names, смешивания источников
  или автоматического чтения inherited environment. Decimal/financial core
  ключей не читает. Возвращаемый `DemoCredentials` имеет `api_key`/`api_secret`,
  но его repr/str и тексты исключений не раскрывают значения. Missing/empty,
  duplicate ключи или неполная explicit пара → ValueError без fallback.
- `python -I -m binance_demo_bot` — offline readiness entrypoint: isolated mode
  исключает подмену package через PYTHONPATH/CWD, затем выполняется проверка
  PYTHONPATH, затем JSON с application ID, own root, `order_send_enabled=false`
  и `financial_core_only=true`. Не читает `.env`, не открывает sockets, не
  создаёт workers и не является production trading service.

Первый batch проверяет части ISO-01/02/03/10; install/import/start дополнительно
проверяются в own venv с poison legacy PYTHONPATH/parent .env и fingerprints
старого source/config. PID/ports/account ownership/resource budgets и все
прочие integration части ISO-04..09 остаются PLANNED; readiness JSON не
означает полного PASS изоляции или разрешения торговли.

Business metric нового приложения — `daily_net_equity_pnl_v1`.
Isolation metrics: число запрещённых import/read/write/process/control
действий, namespace/path violations, shared-token/port collisions, resource
breaches и fingerprints текущих source/config до/после испытания.

Приёмка требует независимого install/start/test/stop нового package; ноль
записей нового app в старые paths, ноль старых process/control mutations;
own environment/state/backups и отсутствие legacy imports/fallback.
Одновременная работа старого бота и failure/restart/rollback нового не меняет
старую конфигурацию, data ownership или запуск. Existing runtime может
обновляться самим старым ботом — это не приписывается новому app.
Проверка resource impact имеет заранее фиксированный baseline/budget.

## Backtest / verification gate

Изоляция проверяется до account execution и automatic learning enablement.
Она не доказывает прибыль; policy changes нового app всё равно требуют
maximum available causal portfolio history и forward/canary фаз 6/7.
Repository Truth Harness — внешний engineering check, не импортируемая
runtime-зависимость нового приложения и не permission key.

Планируемые focused scenarios:

- `ISO-01`: install/start/tests в own venv без `files/` на import path;
- `ISO-02`: cwd/parent `.env`/legacy PYTHONPATH injection rejected;
- `ISO-03`: traversal/symlink/старый DB/model/log/stop path rejected;
- `ISO-04`: два приложения живы, отдельные namespace/token/port/locks;
- `ISO-05`: PID reuse/legacy worker/foreign lock не attach/kill;
- `ISO-06`: crash/restore/rollback нового app сохраняет старый source/config;
- `ISO-07`: old model/ticket/source namespace не разрешает новый release;
- `ISO-08`: same-key/different-key same-account конфликт не скрывается;
- `ISO-09`: собственные resource/aggregate rate breaches ограничивают только app;
- `ISO-10`: credential provisioning не печатает секрет и не меняет старый файл.

P0-A установлен editable в собственное `.venv` (Python 3.14.3), с собственным
`uv.lock`, cache и tzdata. 10 isolation unit-tests и два startup tests GREEN
в общей партии 81 tests. Junction escape проверен реально; injection PYTHONPATH
отвергается при isolated startup. Нет old runtime imports или credential copying.
Полные ISO-01..10 acceptance criteria ещё не закрыты: нет своего OMS/server,
account ownership check, resource trials или process/crash integration.

## Risk / trade-offs and rollback

Независимый package увеличивает начальную реализацию и требует собственного
учёта/контроля качества. В обмен исключает скрытую связанность текущей версии.
При rollback останавливаются собственные new entries/promotions, сохраняются
его ledger/exchange protection и разрешённый reconcile/close. Старое приложение
никогда не используется как fallback executor или владелец новых позиций.
Устанавливать/обновлять/удалять новую среду можно без действий над прежней.
