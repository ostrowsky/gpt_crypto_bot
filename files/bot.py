# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Crypto Trend Bot — Telegram interface.

Меню:
  🔍 Анализ всего списка   → форвард-тест + активные сигналы по всем 100 монетам
  ▶️ Старт мониторинга     → запускает фоновый мониторинг
  ⏹ Стоп мониторинга      → останавливает мониторинг
  📊 Активные сигналы      → открытые позиции + прогнозы
  📋 Список монет          → просмотр / добавление / удаление
  ⚙️ Настройки             → текущие параметры стратегии
"""

from datetime import date as _build_date

BUILD_ID = "menu_build_v3"
BUILD_DATE = _build_date.today().isoformat()

import os
import atexit

LOCK_PATH = "bot.lock"

def _pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True

def _acquire_lock() -> None:
    try:
        fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode("utf-8"))
        os.close(fd)
    except FileExistsError:
        try:
            stale_pid = int(Path(LOCK_PATH).read_text(encoding="utf-8").strip())
        except Exception:
            stale_pid = None
        if stale_pid and not _pid_is_running(stale_pid):
            try:
                os.remove(LOCK_PATH)
            except Exception:
                pass
            fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode("utf-8"))
            os.close(fd)
            return
        raise RuntimeError("Another bot instance is already running (bot.lock exists).")

def _release_lock() -> None:
    try:
        os.remove(LOCK_PATH)
    except Exception:
        pass

_acquire_lock()
atexit.register(_release_lock)

import asyncio
import json
from pathlib import Path
import logging
import re

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, ReplyKeyboardRemove, Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

import config
from monitor import MonitorState, monitoring_loop, load_positions, save_positions
from strategy import market_scan, check_entry_conditions, check_setup_conditions, analyze_coin, fetch_klines, get_entry_mode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
log = logging.getLogger(__name__)

state = MonitorState()
# Фикс: восстанавливаем позиции после рестарта
state.positions = load_positions()
AGENT_POSITIONS_PATH = Path(__file__).resolve().parent / "agent_positions.json"


def _unified_portfolio_limit() -> int:
    return int(
        getattr(
            config,
            "UNIFIED_PORTFOLIO_MAX_POSITIONS",
            getattr(config, "MAX_OPEN_POSITIONS", 6),
        )
        or 0
    )


def build_badge() -> str:
    return f"`v:{BUILD_ID}`  `build:{BUILD_DATE}`"


def signal_mode_label(mode: str) -> str:
    labels = {
        "trend": "📈 Тренд",
        "strong_trend": "💪 Сильный тренд",
        "impulse_speed": "⚡ Быстрое движение",
        "4h_leader_watch": "🧭 4h лидер",
        "retest": "🔄 Ретест EMA20",
        "breakout": "⚡ Пробой флэта",
        "impulse": "🚀 Импульс",
        "impulse_cross": "🚀 Импульс (кросс)",
        "alignment": "🌊 Выравнивание тренда",
    }
    return labels.get(str(mode or "trend"), "📈 Тренд")


# ── Keyboards ─────────────────────────────────────────────────────────────────

def _load_agent_positions() -> dict:
    try:
        raw = json.loads(AGENT_POSITIONS_PATH.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as exc:
        log.warning("Failed to load agent positions from %s: %s", AGENT_POSITIONS_PATH, exc)
        return {}


def _position_to_raw(pos) -> dict:
    fields = (
        "symbol", "tf", "entry_price", "entry_bar", "entry_ts", "entry_ema20",
        "entry_slope", "entry_adx", "entry_rsi", "entry_vol_x",
        "forecast_return_pct", "today_change_pct", "candidate_score_at_entry",
        "ranker_final_score", "ranker_ev", "ranker_top_gainer_prob",
        "four_h_context_score", "four_h_context_label", "predictions",
        "bars_elapsed", "signal_mode", "trail_k", "max_hold_bars",
        "trail_stop",
    )
    raw = {name: getattr(pos, name, None) for name in fields}
    raw["symbol"] = str(raw.get("symbol") or "")
    raw["tf"] = str(raw.get("tf") or "15m")
    return raw


def _main_positions_raw() -> dict:
    return {sym: _position_to_raw(pos) for sym, pos in state.positions.items()}


def _unified_position_rows(*, limit: int | None = None) -> list[dict]:
    from unified_portfolio import ranked_unified_positions

    return ranked_unified_positions(
        _main_positions_raw(),
        _load_agent_positions(),
        limit=limit,
    )


def _agent_prediction_summary(pos: dict) -> str:
    predictions = pos.get("predictions") if isinstance(pos.get("predictions"), dict) else {}
    horizons = list(predictions.keys()) or [3, 5, 10]
    parts = []
    for h in horizons:
        result = predictions.get(str(h), predictions.get(h))
        if result is None:
            icon = "⏳"
        elif result:
            icon = "✅"
        else:
            icon = "❌"
        parts.append(f"T+{h}: {icon}")
    return "  ".join(parts)


def kb_main() -> InlineKeyboardMarkup:
    # Единая кнопка с тремя состояниями:
    #   ▶️ Анализ + Мониторинг  — ни анализа, ни мониторинга нет
    #   🔄 Повторный анализ     — мониторинг уже работает (перезапустить анализ)
    #   ⏹ Стоп мониторинга     — остановить всё
    max_pos = _unified_portfolio_limit()
    pos = min(len(_unified_position_rows(limit=None)), max_pos)
    pos_label = f"{pos}/{max_pos}"
    wl   = len(config.load_watchlist())
    hot  = len(state.hot_coins)
    conf = len([r for r in state.hot_coins if r.today_confirmed])

    if state.running:
        main_btn = InlineKeyboardButton(
            f"⏹ Стоп мониторинга  [{conf} подтв. | {pos} поз.]",
            callback_data="stop_monitor",
        )
        rescan_btn = [InlineKeyboardButton(
            f"🔄 Повторный анализ  [{hot} монет]",
            callback_data="market_scan",
        )]
    else:
        main_btn = InlineKeyboardButton(
            f"▶️ Анализ + Мониторинг  ({wl} монет)",
            callback_data="scan_and_start",
        )
        rescan_btn = [InlineKeyboardButton(
            "🔍 Только анализ",
            callback_data="market_scan",
        )]

    signals_lbl = f"📊 Позиции  [{pos_label}]" if pos else "📊 Позиции"
    list_lbl    = f"📋 Список монет  [{wl}]"

    return InlineKeyboardMarkup([
        [main_btn],
        rescan_btn,
        [InlineKeyboardButton(signals_lbl, callback_data="positions")],
        [InlineKeyboardButton(list_lbl,    callback_data="watchlist")],
        [InlineKeyboardButton("⚙️ Настройки", callback_data="settings")],
    ])


def kb_watchlist() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("➕ Добавить монету", callback_data="add_coin")],
        [InlineKeyboardButton("➖ Удалить монету",  callback_data="del_coin")],
        [InlineKeyboardButton("🔙 Назад",           callback_data="back_main")],
    ])


def kb_back() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Назад", callback_data="back_main")]])


def kb_quick_menu() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        [["📋 Открыть меню", "🙈 Скрыть меню"]],
        resize_keyboard=True,
        is_persistent=True,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_truncate(text: str, max_len: int = 4000) -> str:
    """Обрезает текст до max_len символов по границе строки, не посередине тега."""
    if len(text) <= max_len:
        return text
    # Обрезаем по последнему переносу строки в пределах лимита
    cut = text[:max_len].rfind("\n")
    if cut < max_len // 2:
        cut = max_len  # нет переносов — режем жёстко
    return text[:cut] + "\n…"


async def _send(chat_id: int, text: str, app: Application) -> None:
    await app.bot.send_message(
        chat_id=chat_id, text=_safe_truncate(text), parse_mode=ParseMode.MARKDOWN,
    )


async def _answer_callback_fast(query) -> None:
    try:
        await asyncio.wait_for(query.answer(), timeout=1.0)
    except Exception:
        pass


async def _edit_or_send(query, app: Application, text: str, *, parse_mode=None, reply_markup=None) -> None:
    try:
        await asyncio.wait_for(
            query.edit_message_text(
                text=_safe_truncate(text),
                parse_mode=parse_mode,
                reply_markup=reply_markup,
            ),
            timeout=2.5,
        )
        return
    except Exception as exc:
        log.warning("edit_message_text failed for action=%s: %s", getattr(query, "data", ""), exc)
    await asyncio.wait_for(
        app.bot.send_message(
            chat_id=query.message.chat_id,
            text=_safe_truncate(text),
            parse_mode=parse_mode,
            reply_markup=reply_markup,
        ),
        timeout=5.0,
    )


def _early_signal_line(r) -> str:
    """Компактная строка для секции ⚡ — сигнал активен но не подтверждён."""
    acc_parts = "  ".join(
        str(r.today_accuracy[h])
        for h in config.FORWARD_BARS
        if h in r.today_accuracy and r.today_accuracy[h].total > 0
    ) or "нет оценок"
    return (
        f"⚡ *{r.symbol}* `[{r.tf}]`  "
        f"сигналов сегодня: {r.today_signals}  {acc_parts}\n"
        f"   Цена: `{r.current_price:.6g}`  "
        f"slope:`{r.current_slope:+.2f}%`  "
        f"RSI:`{r.current_rsi:.1f}`  "
        f"ADX:`{r.current_adx:.1f}`  "
        f"vol×:`{r.current_vol_x:.2f}`"
    )


def _format_early_watch(now_ms: int, window_ms: int = 3600_000) -> str:
    """Return EARLY WATCH section for last N ms from MonitorState."""
    items = []
    msgs = getattr(state, "early_warn_msgs", {}) or {}
    for key, val in msgs.items():
        try:
            ts, txt = val
        except Exception:
            continue
        if now_ms - int(ts) <= window_ms:
            items.append((int(ts), txt))
    items.sort(key=lambda x: x[0], reverse=True)
    if not items:
        return ""
    # show up to 5
    lines = ["⚠️ *EARLY WATCH (последний час):*"]
    for _, t in items[:5]:
        lines.append(t)
    if len(items) > 5:
        lines.append(f"_...и ещё {len(items)-5} предупреждений_")
    # EARLY WATCH section
    try:
        now_ms = int(time.time()*1000)
        ew = _format_early_watch(now_ms)
        if ew:
            lines.append("\n"+ew)
    except Exception:
        pass
    return "\n".join(lines)

def _format_analysis_result(in_play, skipped, title: str, scan: bool = False) -> str:
    lines = [title, ""]

    if in_play:
        active  = [r for r in in_play if r.signal_now]
        waiting = [r for r in in_play if not r.signal_now]

        header = f"✅ *Подтверждено сегодня — {len(in_play)} монет:*"
        if active:
            header += f"\n🟢 *Сигнал активен прямо сейчас: {len(active)}*"
        lines.append(header)

        for r in active:
            lines.append(r.summary())

        if waiting:
            lines.append(f"\n⏸ *Стратегия подтверждена, ждут сигнала: {len(waiting)}*")
            for r in waiting:
                lines.append(r.summary())
    else:
        lines.append(
            "❌ Ни одна монета не прошла порог точности.\n"
            f"Порог точности: {config.MIN_ACCURACY}%  |  T+3 ≥ {config.TODAY_T3_MIN}%  |  T+10 ≥ {config.TODAY_T10_MIN}%\n"
            f"Мин. сигналов сегодня: {config.TODAY_MIN_SIGNALS}"
        )

    # ── ⚡ BUY активен, но форвард-тест не подтверждён ─────────────────────────
    early = sorted(
        [r for r in skipped if r.signal_now],
        key=lambda r: r.today_signals, reverse=True,
    )
    if early:
        lines.append(
            f"\n⚡ *BUY активен, не подтверждены ({len(early)}):*\n"
            f"_Форвард-тест не накопил данных — повышенный риск_"
        )
        for r in early[:6]:
            lines.append(_early_signal_line(r))
        if len(early) > 6:
            lines.append(f"_...и ещё {len(early) - 6} с активным BUY_")

    # ── 🟡 SETUP — тренд зарождается, до BUY не хватает 1 фильтра ───────────
    setup_list = sorted(
        [r for r in skipped if r.setup_now and not r.signal_now],
        key=lambda r: r.today_signals, reverse=True,
    )
    if setup_list:
        lines.append(f"\n🟡 *SETUP — структура складывается ({len(setup_list)}):*")
        for r in setup_list[:6]:
            lines.append(
                f"🟡 *{r.symbol}* `[{r.tf}]`  "
                f"`{r.current_price:.6g}`  "
                f"slope:`{r.current_slope:+.2f}%`  "
                f"RSI:`{r.current_rsi:.1f}`  "
                f"vol×:`{r.current_vol_x:.2f}`\n"
                + "   _" + (r.setup_reason
                    .replace("[", "\\[").replace("]", "\\]")
                    .replace("_", "\\_").replace("(", "\\(").replace(")", "\\)")
                ) + "_"
            )
        if len(setup_list) > 6:
            lines.append(f"_...и ещё {len(setup_list) - 6} в SETUP_")

    # ── Пропущенные без сигнала ───────────────────────────────────────────────
    cold = [r for r in skipped if not r.signal_now and not r.setup_now]
    if cold:
        lines.append(f"\n⏭ *Пропущено ({len(cold)}):*")
        for r in cold[:3]:
            lines.append(r.summary())
        if len(cold) > 3:
            lines.append(f"_...и ещё {len(cold) - 3} монет не прошли порог_")

    # EARLY WATCH section
    try:
        now_ms = int(time.time()*1000)
        ew = _format_early_watch(now_ms)
        if ew:
            lines.append("\n"+ew)
    except Exception:
        pass
    return "\n".join(lines)


# ── Handlers ──────────────────────────────────────────────────────────────────

# Хранилище chat_id для авто-уведомлений и реанализа
_known_chat_ids: set[int] = set()

try:
    _chat_ids_file = Path(".chat_ids")
    if _chat_ids_file.exists():
        import json as _json
        _known_chat_ids = set(_json.loads(_chat_ids_file.read_text()))
except Exception:
    pass


def _save_chat_id(chat_id: int) -> None:
    _known_chat_ids.add(chat_id)
    try:
        import json as _json
        Path(".chat_ids").write_text(_json.dumps(list(_known_chat_ids)))
    except Exception:
        pass


def _make_broadcast_send(app: "Application"):
    """
    Возвращает функцию send которая рассылает сообщение всем известным chat_id.
    РСЃРїРѕР»СЊР·СѓРµС‚СЃСЏ РґР»СЏ monitoring_loop Р·Р°РїСѓС‰РµРЅРЅРѕРіРѕ РёР· Р°РІС‚Рѕ-СЂРµР°РЅР°Р»РёР·Р°.
    """
    async def _broadcast(text: str) -> None:
        for cid in list(_known_chat_ids):
            try:
                await _send(cid, text, app)
            except Exception as e:
                log.warning("broadcast send failed for %s: %s", cid, e)
        # Сохраняем chat_ids на диск при каждой отправке
        try:
            import json as _json
            Path(".chat_ids").write_text(_json.dumps(list(_known_chat_ids)))
        except Exception:
            pass
    return _broadcast



def _update_hot_coins(state, in_play, skipped) -> None:
    """
    Обновляет state.hot_coins:
    - confirmed монеты (in_play) — мониторятся всегда
    - НЕ-confirmed монеты с signal_now=True — мониторятся с пометкой риска
    
    Это главный фикс: раньше монеты типа TRXUSDT (signal_now=True, confirmed=False)
    полностью игнорировались мониторингом. Теперь они включаются.
    """
    # Confirmed всегда в списке
    hot = list(in_play)
    
    # Добавляем не-confirmed с активным сигналом прямо сейчас
    already = {r.symbol for r in hot}
    for r in skipped:
        if r.signal_now and r.symbol not in already:
            hot.append(r)
    
    state.hot_coins = hot


async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.message.chat_id
    _save_chat_id(chat_id)
    wl     = config.load_watchlist()
    status = "▶️ запущен" if state.running else "⏹ остановлен"
    text   = (
        f"👋 *Crypto Trend Bot*\n\n"
        f"{build_badge()}\n\n"
        f"Мониторинг: {status}\n"
        f"Монет в списке: *{len(wl)}*\n"
        f"Монет «в игре» сегодня: *{len(state.hot_coins)}*\n"
        f"Открытых сигналов: *{len(state.positions)}*"
    )
    await update.message.reply_text(
        "Кнопки меню включены.",
        reply_markup=kb_quick_menu(),
    )
    await update.message.reply_text(
        text, parse_mode=ParseMode.MARKDOWN, reply_markup=kb_main(),
    )


async def cmd_positions(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not state.positions:
        txt = "📊 Активных позиций нет."
    else:
        lines = [f"📊 Открытых позиций: {len(state.positions)}"]
        for sym, pos in state.positions.items():
            lines.append(
                f"- {sym} [{pos.tf}] {signal_mode_label(getattr(pos, 'signal_mode', 'trend'))} "
                f"entry={pos.entry_price:.6g} bars={pos.bars_elapsed}"
            )
        txt = "\n".join(lines)
    await update.message.reply_text(txt, reply_markup=kb_main())


async def btn(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    query   = update.callback_query
    asyncio.create_task(_answer_callback_fast(query))
    action  = query.data
    chat_id = query.message.chat_id
    log.info("MENU CALLBACK action=%s chat_id=%s", action, chat_id)

    # ── 🌅 Анализ на день ─────────────────────────────────────────────────────
    if action == "market_scan":
        n = len(config.load_watchlist())
        rescan_note = " (мониторинг продолжается)" if state.running else ""
        await _edit_or_send(
            query,
            ctx.application,
            f"🔍 Анализирую список — *{n}* монет{rescan_note}...\n"
            f"Форвард-тест на данных сегодняшнего дня.\n"
            f"Займёт ~1–2 минуты.",
            parse_mode=ParseMode.MARKDOWN,
        )
        in_play, skipped = await market_scan()
        _update_hot_coins(state, in_play, skipped)

        from datetime import datetime, timezone
        now_str    = datetime.now(timezone.utc).strftime("%H:%M UTC")
        conf_count = len([r for r in in_play if r.today_confirmed])
        mon_note   = f"\n▶️ Мониторинг продолжается | Подтверждено: *{conf_count}*" if state.running else ""
        text = _format_analysis_result(
            in_play, skipped,
            f"🔍 *Анализ завершён* — {n} монет  _{now_str}_{mon_note}",
        )
        await _edit_or_send(
            query,
            ctx.application,
            _safe_truncate(text), parse_mode=ParseMode.MARKDOWN, reply_markup=kb_main(),
        )

    # ── ▶️ Анализ + Мониторинг (единая кнопка) ───────────────────────────────
    elif action == "scan_and_start":
        if state.running:
            # query.answer() уже был вызван выше — дублировать нельзя
            await _edit_or_send(
            query,
            ctx.application,
                "▶️ Мониторинг уже запущен.",
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=kb_main(),
            )
            return
        n = len(config.load_watchlist())
        await _edit_or_send(
            query,
            ctx.application,
            f"🔍 Анализирую список — *{n}* монет...\n"
            f"Форвард-тест на данных сегодняшнего дня.\n"
            f"Займёт ~1–2 минуты, затем мониторинг запустится автоматически.",
            parse_mode=ParseMode.MARKDOWN,
        )
        in_play, skipped = await market_scan()
        _update_hot_coins(state, in_play, skipped)

        # Автоматически стартуем мониторинг после анализа
        state.running = True

        async def send_msg_sas(text: str) -> None:
            # Отправляем в чат где нажата кнопка + всем остальным подписчикам
            sent_to = set()
            for cid in list(_known_chat_ids):
                try:
                    await _send(cid, text, ctx.application)
                    sent_to.add(cid)
                except Exception:
                    pass
            # Если chat_id почему-то не в _known_chat_ids — шлём отдельно
            if chat_id not in sent_to:
                try:
                    await _send(chat_id, text, ctx.application)
                except Exception:
                    pass

        state.task = asyncio.create_task(monitoring_loop(state, send_msg_sas))

        from datetime import datetime, timezone
        now_str    = datetime.now(timezone.utc).strftime("%H:%M UTC")
        conf_count = len([r for r in in_play if r.today_confirmed])
        text = _format_analysis_result(
            in_play, skipped,
            f"🔍 *Анализ завершён* — {n} монет  _{now_str}_\n"
            f"▶️ *Мониторинг запущен автоматически* | Подтверждено: *{conf_count}*",
        )
        await _edit_or_send(
            query,
            ctx.application,
            _safe_truncate(text), parse_mode=ParseMode.MARKDOWN, reply_markup=kb_main(),
        )

    # ── ⏹ Стоп мониторинга ───────────────────────────────────────────────────
    elif action == "stop_monitor":
        state.running = False
        if state.task:
            state.task.cancel()
            state.task = None
        await _edit_or_send(
            query,
            ctx.application,
            f"⏹ *Мониторинг остановлен.*\n"
            f"Открытых сигналов: {len(state.positions)}",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=kb_main(),
        )

    # ── 📊 Активные сигналы ───────────────────────────────────────────────────
    elif action == "positions":
        max_pos = _unified_portfolio_limit()
        rows_all = _unified_position_rows(limit=None)
        rows = rows_all[:max_pos]
        if not rows:
            txt = f"📊 Активных позиций нет.  <i>(лимит: {max_pos})</i>"
        else:
            import html as _html
            from monitor import _get_coin_group
            MAX_LEN  = 4000
            filled = min(len(rows), max_pos)
            port_bar = "█" * filled + "░" * max(0, max_pos - filled)
            lines = [
                f"📊 <b>Единый портфель: {len(rows)}/{max_pos}</b>\n",
                f"<b>Top-{max_pos} по перспективности:</b> <code>{port_bar}</code>\n",
            ]
            if len(rows_all) > len(rows):
                lines.append(f"<i>Скрыто слабее лимита: {len(rows_all) - len(rows)}</i>\n")
            shown = 0
            for idx, row in enumerate(rows, start=1):
                pos = row["position"]
                source = str(row["source"])
                sym = str(row["symbol"])
                tf = str(pos.get("tf") or "15m")
                mode = str(pos.get("signal_mode") or "trend")
                entry = float(pos.get("entry_price") or 0.0)
                bars = int(pos.get("bars_elapsed") or 0)
                slope = float(pos.get("entry_slope") or 0.0)
                adx = float(pos.get("entry_adx") or 0.0)
                score = float(row.get("score") or 0.0)
                forecast_return = float(pos.get("forecast_return_pct") or 0.0)
                today_change = float(pos.get("today_change_pct") or 0.0)
                four_h_score = float(pos.get("four_h_context_score") or 0.0)
                four_h_label = str(pos.get("four_h_context_label") or "")
                scan_icon = " 🔍" if any(
                    r.symbol == sym and r.from_scan for r in state.hot_coins
                ) else ""
                grp = _get_coin_group(sym)
                grp_str = f"  <i>[{_html.escape(grp)}]</i>" if grp else ""

                # EV из hot_coins если доступно
                ev_line = ""
                coin_report = next(
                    (r for r in state.hot_coins if r.symbol == sym), None
                )
                if coin_report and coin_report.today_accuracy:
                    ev_parts = []
                    for h, fa in sorted(coin_report.today_accuracy.items()):
                        expected_return = getattr(fa, "expected_return", None)
                        if fa.total > 0 and expected_return is not None:
                            ev_icon = "▲" if expected_return > 0 else "▼"
                            ev_parts.append(f"T+{h}:{fa.pct:.0f}%{ev_icon}{expected_return:+.2f}%")
                        elif fa.total > 0:
                            ev_parts.append(f"T+{h}:{fa.pct:.0f}%")
                    if ev_parts:
                        ev_line = f"  📊 {' '.join(ev_parts)}\n"
                four_h_line = ""
                if four_h_label:
                    four_h_line = f"  🕓 4h: <code>{four_h_score:+.1f}</code> {_html.escape(four_h_label)}\n"

                block = (
                    f"<b>{idx}. {_html.escape(sym)}</b>{scan_icon} <i>[{source}]</i>{grp_str}  "
                    f"<code>[{_html.escape(tf)}]</code>\n"
                    f"  🧭 {_html.escape(signal_mode_label(mode))}  "
                    f"score <code>{score:.1f}</code>  "
                    f"📈 slope <code>{slope:+.2f}%</code>  "
                    f"💪 ADX <code>{adx:.1f}</code>\n"
                    f"  💰 Вход: <code>{entry:.6g}</code>  "
                    f"⏱ {bars}б  "
                    f"forecast <code>{forecast_return:+.2f}%</code> today <code>{today_change:+.2f}%</code>\n"
                    + four_h_line
                    + ev_line
                    + f"  🎯 {_agent_prediction_summary(pos)}\n"
                )
                current_len = sum(len(l) for l in lines)
                if current_len + len(block) > MAX_LEN:
                    remaining = len(rows) - shown
                    lines.append(f"\n<i>...и ещё {remaining} позиций</i>")
                    break
                lines.append(block)
                shown += 1
            txt = "\n".join(lines)
        # Positions are easier to miss when Telegram edits an older menu message.
        # Send them as a fresh message so the button always has visible feedback.
        await asyncio.wait_for(
            ctx.application.bot.send_message(
                chat_id=chat_id,
                text=_safe_truncate(txt),
                parse_mode=ParseMode.HTML,
                reply_markup=kb_main(),
            ),
            timeout=5.0,
        )

    # ── 📋 Список монет ───────────────────────────────────────────────────────
    elif action == "watchlist":
        wl   = config.load_watchlist()
        rows = [wl[i:i+5] for i in range(0, len(wl), 5)]
        grid = "\n".join("  ".join(f"`{s}`" for s in row) for row in rows)
        txt  = f"📋 *Ваш список монет* ({len(wl)} шт.):\n\n{grid}"
        await _edit_or_send(
            query,
            ctx.application,
            _safe_truncate(txt), parse_mode=ParseMode.MARKDOWN, reply_markup=kb_watchlist(),
        )

    elif action == "add_coin":
        ctx.user_data["awaiting"] = "add_coin"
        await _edit_or_send(
            query,
            ctx.application,
            "Введите тикер монеты, например `SOLUSDT`:",
            parse_mode=ParseMode.MARKDOWN,
        )

    elif action == "del_coin":
        ctx.user_data["awaiting"] = "del_coin"
        wl = config.load_watchlist()
        await _edit_or_send(
            query,
            ctx.application,
            "Введите тикер для удаления.\n\nСейчас в списке:\n"
            + "  ".join(f"`{s}`" for s in wl[:20])
            + (" ..." if len(wl) > 20 else ""),
            parse_mode=ParseMode.MARKDOWN,
        )

    # ── ⚙️ Настройки ──────────────────────────────────────────────────────────
    elif action == "settings":
        tf_str = ", ".join(config.TIMEFRAMES)
        fb_str = ", ".join(f"T+{h}" for h in config.FORWARD_BARS)
        txt = (
            f"⚙️ *Текущие настройки*\n\n"
            f"*Таймфреймы:* `{tf_str}`\n"
            f"*Горизонты прогноза:* `{fb_str}`\n"
            f"*Порог точности:* `{config.MIN_ACCURACY}%`\n"
            f"*Дневное подтверждение:*\n"
            f"  Мин. сигналов сегодня: `{config.TODAY_MIN_SIGNALS}`\n"
            f"  T+3 ≥ `{config.TODAY_T3_MIN}%`  |  T+10 ≥ `{config.TODAY_T10_MIN}%`\n"
            f"  Лучший горизонт ≥ `{config.MIN_ACCURACY}%`\n\n"
            f"*Условия входа:*\n"
            f"  Цена > EMA20 > EMA50\n"
            f"  EMA20 наклон ≥ `{config.EMA_SLOPE_MIN}%` за {config.SLOPE_LOOKBACK} баров\n"
            f"  ADX ≥ `{config.ADX_MIN}` и ADX > SMA(ADX,{config.ADX_SMA_PERIOD})\n"
            f"  Объём ≥ `{config.VOL_MULT}×` среднего\n"
            f"  RSI: `{config.RSI_BUY_LO}` – `{config.RSI_BUY_HI}`\n"
            f"  MACD гистограмма > 0\n"
            f"  Рост от дна дня < `{config.DAILY_RANGE_MAX}%`\n\n"
            f"*Условия выхода:*\n"
            f"  ATR-трейлинг (×`{config.ATR_TRAIL_K}` ATR / сильный тренд ×`{getattr(config,'ATR_TRAIL_K_STRONG', config.ATR_TRAIL_K + 0.5):.1f}`)\n"
            f"  2 закрытия подряд ниже EMA20 — ранний разворот _(новое)_\n"
            f"  Одиночное закрытие ниже EMA20 — страховка\n"
            f"  RSI > `{config.RSI_OVERBOUGHT}`\n"
            f"  EMA20 slope < 0\n"
            f"  ADX < `{config.ADX_DROP_RATIO*100:.0f}%` × ADX[{config.ADX_GROW_BARS} бара назад]\n"
            f"  Лимит: `{config.MAX_HOLD_BARS}` баров\n\n"
            f"*ADX-bypass:* при ADX ≥ `{getattr(config,'ADX_SMA_BYPASS',35.0):.0f}` фильтр ADX>SMA пропускается\n"
            f"⚠️ MACD-предупреждение: падает `{config.MACDWARN_BARS}` бара подряд\n"
            f"🔄 Авто-реанализ: каждые `{config.AUTO_REANALYZE_SEC//3600}ч`\n\n"
            f"? ???????? ??????: `{config.POLL_SEC}?`\n\n"
            f"_Для изменения — отредактируйте config.py_"
        )
        await _edit_or_send(
            query,
            ctx.application,
            txt, parse_mode=ParseMode.MARKDOWN, reply_markup=kb_back(),
        )

    elif action == "back_main":
        wl = config.load_watchlist()
        await _edit_or_send(
            query,
            ctx.application,
            f"{build_badge()}\n\n"
            f"Монет в списке: *{len(wl)}*  |  В игре: *{len(state.hot_coins)}*  |  "
            f"Сигналов: *{len(state.positions)}*",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=kb_main(),
        )


# ── Text input (add / remove coin) ────────────────────────────────────────────

async def cmd_why(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /why SYMBOL — показывает почему нет сигнала по монете прямо сейчас."""
    parts = (update.message.text or "").strip().split()
    if len(parts) < 2:
        await update.message.reply_text(
            "РСЃРїРѕР»СЊР·РѕРІР°РЅРёРµ: `/why SYMBOL`\nРџСЂРёРјРµСЂ: `/why TONUSDT`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    symbol = parts[1].upper()
    if not symbol.endswith("USDT"):
        symbol += "USDT"

    await update.message.reply_text(
        f"🔍 Проверяю *{symbol}*...", parse_mode=ParseMode.MARKDOWN
    )

    import aiohttp as _aiohttp
    import numpy as np
    from indicators import compute_features
    from strategy import (
        check_retest_conditions, check_breakout_conditions,
        check_impulse_conditions, check_alignment_conditions,
    )

    results = []
    async with _aiohttp.ClientSession() as session:
        for tf in config.TIMEFRAMES:
            # РСЃРїРѕР»СЊР·СѓРµРј LIVE_LIMIT вЂ” С‚Рµ Р¶Рµ РґР°РЅРЅС‹Рµ С‡С‚Рѕ Рё Сѓ РјРѕРЅРёС‚РѕСЂРёРЅРіР°
            data = await fetch_klines(session, symbol, tf, limit=config.LIVE_LIMIT)
            if data is None:
                results.append(f"`{tf}`: ❌ нет данных с Binance")
                continue

            c    = data["c"].astype(float)
            feat = compute_features(data["o"], data["h"], data["l"], c, data["v"])
            i    = len(c) - 2

            ef    = float(feat["ema_fast"][i])
            es    = float(feat["ema_slow"][i])
            slp   = float(feat["slope"][i])
            rsi   = float(feat["rsi"][i])
            adx   = float(feat["adx"][i])
            adx_s = float(feat["adx_sma"][i]) if np.isfinite(feat["adx_sma"][i]) else 0.0
            vx    = float(feat["vol_x"][i])
            dr    = float(feat["daily_range_pct"][i]) if np.isfinite(feat["daily_range_pct"][i]) else 0.0
            price = float(c[i])

            buy_ok,  buy_r  = check_entry_conditions(feat, i, c)
            ret_ok,  ret_r  = check_retest_conditions(feat, i)
            brk_ok,  brk_r  = check_breakout_conditions(feat, i)
            imp_ok,  imp_r  = check_impulse_conditions(feat, i)
            aln_ok,  aln_r  = check_alignment_conditions(feat, i)
            stp_ok,  stp_r, _ = check_setup_conditions(feat, i, c)

            any_sig = buy_ok or ret_ok or brk_ok or imp_ok or aln_ok

            def _fmt(ok, reason, label):
                icon = "🟢" if ok else "🔴"
                return f"{icon} {label}" + ("" if ok else f": {reason[:80]}")

            lines = [
                f"*{tf}*  `{price:.6g}`  EMA20:`{ef:.4g}`  EMA50:`{es:.4g}`",
                f"  slope:`{slp:+.2f}%`  ADX:`{adx:.1f}`(sma:{adx_s:.1f})  RSI:`{rsi:.1f}`  vol×:`{vx:.2f}`  dr:`{dr:.1f}%`",
                f"  {'вњ… РЎРР“РќРђР› Р•РЎРўР¬' if any_sig else 'в›” СЃРёРіРЅР°Р»Р° РЅРµС‚'}",
                f"  {_fmt(buy_ok, buy_r, 'BUY')}",
                f"  {_fmt(ret_ok, ret_r, 'RETEST')}",
                f"  {_fmt(brk_ok, brk_r, 'BREAKOUT')}",
                f"  {_fmt(imp_ok, imp_r, 'IMPULSE')}",
                f"  {_fmt(aln_ok, aln_r, 'ALIGNMENT')}",
            ]
            if not any_sig:
                lines.append(f"  ⬜ SETUP: {'есть' if stp_ok else stp_r[:80]}")
            results.append("\n".join(lines))

    text = f"🔍 *{symbol}* (данные мониторинга, {config.LIVE_LIMIT} баров)\n\n" + "\n\n".join(results)
    await update.message.reply_text(_safe_truncate(text), parse_mode=ParseMode.MARKDOWN)


async def cmd_test(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /test — диагностика состояния бота и мониторинга."""
    from datetime import datetime, timezone
    chat_id = update.message.chat_id

    task_alive = False
    task_status = "⏹ остановлен"
    if state.task is not None:
        if state.task.done():
            exc = state.task.exception() if not state.task.cancelled() else None
            task_status = f"💀 зомби (упал: {exc})" if exc else "💀 зомби (отменён)"
            # Реанимируем
            if state.running:
                state.task = asyncio.create_task(
                    monitoring_loop(state, _make_broadcast_send(ctx.application))
                )
                task_status += " → 🔄 перезапущен"
        else:
            task_alive = True
            task_status = "▶️ работает"
    elif state.running:
        task_status = "⚠️ running=True но task=None"

    import time as _time
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    now_ms  = int(_time.time() * 1000)

    # Позиции
    pos_lines = ""
    if state.positions:
        lines = []
        for sym, pos in state.positions.items():
            bars_in = (now_ms - pos.entry_ts) // (
                15 * 60 * 1000 if pos.tf == "15m" else 60 * 60 * 1000
            )
            lines.append(f"  {sym} [{pos.tf}] {pos.signal_mode} @{pos.entry_price:.6g} | {bars_in} баров")
        pos_lines = "\n<b>💼 Открытые позиции:</b>\n" + "\n".join(lines)
    else:
        pos_lines = "\n💼 Открытых позиций: 0"

    # Кулдауны
    cd_lines = ""
    if state.cooldowns:
        active_cd = {s: u for s, u in state.cooldowns.items() if u > now_ms}
        if active_cd:
            lines = []
            for sym, until_ms in sorted(active_cd.items(), key=lambda x: x[1]):
                left_min = max(0, (until_ms - now_ms) // 60000)
                left_h   = left_min // 60
                left_m   = left_min % 60
                time_str = f"{left_h}ч {left_m}м" if left_h else f"{left_m}м"
                lines.append(f"  {sym}: ещё {time_str}")
            cd_lines = f"\n\n<b>⏳ Кулдаун ({len(active_cd)} монет):</b>\n" + "\n".join(lines)
        else:
            cd_lines = "\n\n⏳ Кулдаунов нет"
    else:
        cd_lines = "\n\n⏳ Кулдаунов нет"

    # HTML — не зависит от [] () * _ в exception-тексте и chat_id
    text = (
        f"✅ <b>Тестовое сообщение</b>\n\n"
        f"🕐 Время UTC: <code>{now_str}</code>\n"
        f"📡 Мониторинг: {task_status}\n"
        f"👀 Монет в слежке: {len(state.hot_coins)}"
        f"{pos_lines}"
        f"{cd_lines}\n\n"
        f"рџ“¬ РР·РІРµСЃС‚РЅС‹С… С‡Р°С‚РѕРІ: <code>{list(_known_chat_ids)}</code>\n"
        f"🔑 Этот chat_id: <code>{chat_id}</code>"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)


async def text_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    incoming_text = (update.message.text or "").strip()
    chat_id = update.effective_chat.id if update.effective_chat else None
    log.info("MENU TEXT chat_id=%s text=%r", chat_id, incoming_text)
    lower_text = incoming_text.lower()
    if incoming_text == "📋 Открыть меню":
        await cmd_start(update, ctx)
        return
    if "открыть" in lower_text and "меню" in lower_text:
        await cmd_start(update, ctx)
        return
    if incoming_text == "🙈 Скрыть меню":
        await update.message.reply_text(
            "Кнопки меню скрыты. Чтобы вернуть их, отправьте /menu.",
            reply_markup=ReplyKeyboardRemove(),
        )
        return
    if "скрыть" in lower_text and "меню" in lower_text:
        await update.message.reply_text(
            "Кнопки меню скрыты. Чтобы вернуть их, отправьте /menu.",
            reply_markup=ReplyKeyboardRemove(),
        )
        return

    awaiting = ctx.user_data.get("awaiting")
    if not awaiting:
        await cmd_start(update, ctx)
        return

    ticker = update.message.text.strip().upper()
    wl     = config.load_watchlist()
    if awaiting in {"add_coin", "del_coin"} and not re.fullmatch(r"[A-Z0-9]+USDT", ticker):
        await update.message.reply_text(
            "⚠️ Тикер должен быть в формате `SYMBOLUSDT`, например `SOLUSDT`.",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=kb_main(),
        )
        ctx.user_data.pop("awaiting", None)
        return

    if awaiting == "add_coin":
        if ticker in wl:
            await update.message.reply_text(
                f"⚠️ `{ticker}` уже есть в списке.", parse_mode=ParseMode.MARKDOWN,
            )
        else:
            wl.append(ticker)
            config.save_watchlist(wl)
            await update.message.reply_text(
                f"✅ `{ticker}` добавлен. Всего: {len(wl)}",
                parse_mode=ParseMode.MARKDOWN, reply_markup=kb_main(),
            )

    elif awaiting == "del_coin":
        if ticker not in wl:
            await update.message.reply_text(
                f"⚠️ `{ticker}` не найден.", parse_mode=ParseMode.MARKDOWN,
            )
        else:
            wl.remove(ticker)
            config.save_watchlist(wl)
            state.hot_coins = [r for r in state.hot_coins if r.symbol != ticker]
            await update.message.reply_text(
                f"✅ `{ticker}` удалён. Осталось: {len(wl)}",
                parse_mode=ParseMode.MARKDOWN, reply_markup=kb_main(),
            )

    ctx.user_data.pop("awaiting", None)


# ── Entry point ───────────────────────────────────────────────────────────────

def _ensure_positions_monitored(state) -> None:
    """
    Гарантирует что монеты с открытыми позициями всегда в hot_coins.
    Фикс 15.03.2026: реанализ больше не закрывает открытые позиции.
    Позиция выходит только по техническим условиям (ATR/EMA20/RSI/hold).
    """
    from strategy import CoinReport
    hot_syms = {r.symbol for r in state.hot_coins}
    for sym, pos in list(state.positions.items()):
        if sym not in hot_syms:
            from strategy import CoinReport
            dummy = CoinReport(
                symbol=sym, tf=pos.tf,
                today_signals=0, today_confirmed=True,
                signal_now=False, today_accuracy={},
                best_horizon=0,
                best_accuracy=0.0,
                note="удерживаем позицию (exit guard)",
                in_play=True,
            )
            state.hot_coins.append(dummy)
            log.info("_ensure_positions_monitored: keeping %s [%s] in hot_coins", sym, pos.tf)



async def _auto_reanalyze(app: Application) -> None:
    """Фоновая задача: пересчитывает список монет каждые AUTO_REANALYZE_SEC секунд."""
    if config.AUTO_REANALYZE_SEC <= 0:
        return
    while True:
        await asyncio.sleep(config.AUTO_REANALYZE_SEC)
        try:
            from strategy import market_scan, check_entry_conditions, check_setup_conditions, analyze_coin, fetch_klines, get_entry_mode
            from datetime import datetime, timezone
            log.info("Auto-reanalyze started")
            in_play, skipped = await market_scan()
            _update_hot_coins(state, in_play, skipped)
            # Фикс 15.03.2026: монеты с позицией остаются в мониторинге
            _ensure_positions_monitored(state)
            now_str = datetime.now(timezone.utc).strftime("%H:%M UTC")
            n = len(config.load_watchlist())

            # ── Авто-старт мониторинга если он не работает (после перезапуска бота) ──
            _task_dead = state.task is not None and state.task.done()
            if _task_dead:
                exc = state.task.exception() if not state.task.cancelled() else None
                log.warning("Auto-reanalyze: zombie task detected (exc=%s) — restarting", exc)
                state.task = None
                state.running = False

            if in_play and (not state.running or _task_dead):
                state.running = True
                state.task = asyncio.create_task(
                    monitoring_loop(state, _make_broadcast_send(app))
                )
                mon_note = f"\n▶️ *Мониторинг запущен автоматически*"
            elif not in_play and state.running and not state.positions:
                # Нет подтверждённых монет и нет открытых позиций — останавливаем
                state.running = False
                if state.task:
                    state.task.cancel()
                    state.task = None
                mon_note = "\n⏹ Мониторинг приостановлен (нет подтверждённых)"
            else:
                mon_note = ""

            msg = (
                f"🔄 *Авто-реанализ* _{now_str}_\n"
                f"Пересчитано {n} монет.\n"
                f"Подтверждено: *{len(in_play)}*"
                f"{mon_note}"
            )
            for cid in list(_known_chat_ids):
                try:
                    await _send(cid, msg, app)
                except Exception:
                    pass
        except Exception as e:
            log.error("Auto-reanalyze error: %s", e)


async def _post_init(app: Application) -> None:
    """При старте: уведомляет пользователей и автоматически запускает анализ+мониторинг."""
    asyncio.create_task(_auto_reanalyze(app))

    # ML: запускаем фоновый сборщик данных (каждые 15 минут обновляет ml_dataset.jsonl)
    # Важно: data_collector импортируется здесь чтобы не создавать циклических импортов
    if getattr(config, "BOT_ENABLE_DATA_COLLECTOR", False):
        try:
            import data_collector as _dc
            asyncio.create_task(_dc.run_forever(app))
            log.info("DataCollector task started")
        except Exception as _dc_err:
            log.error("DataCollector failed to start: %s", _dc_err)
    else:
        log.info("DataCollector disabled in Telegram bot process")

    asyncio.create_task(_startup_scan_and_resume(app))


async def _startup_scan_and_resume(app: Application) -> None:
    if not _known_chat_ids:
        return

    wl = config.load_watchlist()
    for cid in list(_known_chat_ids):
        try:
            await app.bot.send_message(
                chat_id=cid,
                text="Кнопки меню включены.",
                reply_markup=kb_quick_menu(),
            )
            await app.bot.send_message(
                chat_id=cid,
                text=(
                    f"🤖 *Бот перезапущен*\n\n"
                    f"{build_badge()}\n\n"
                    f"Монет в списке: *{len(wl)}*\n"
                    f"Меню уже доступно. Фоновый анализ запускаю отдельно."
                ),
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=kb_main(),
            )
        except Exception as e:
            log.warning("startup notify failed for %s: %s", cid, e)

    if not getattr(config, "BOT_STARTUP_AUTO_SCAN_ENABLED", False):
        log.info("Startup auto scan disabled; menu remains interactive")
        return

    try:
        from strategy import market_scan
        in_play, skipped = await market_scan()
        _update_hot_coins(state, in_play, skipped)

        if not state.running:
            state.task = asyncio.create_task(monitoring_loop(state, _make_broadcast_send(app)))
            state.running = True
        mon_line = f"▶️ Мониторинг запущен | Монет: {len(state.hot_coins)}"

        from datetime import datetime, timezone
        now_str = datetime.now(timezone.utc).strftime("%H:%M UTC")
        for cid in list(_known_chat_ids):
            try:
                await app.bot.send_message(
                    chat_id=cid,
                    text=f"🔍 Анализ завершён — {len(wl)} монет  {now_str}\n{build_badge()}\n{mon_line}",
                    parse_mode=ParseMode.MARKDOWN,
                    reply_markup=kb_main(),
                )
            except Exception as e:
                log.warning("startup result notify failed for %s: %s", cid, e)
    except Exception as e:
        log.error("startup auto-analysis failed: %s", e)
        for cid in list(_known_chat_ids):
            try:
                await app.bot.send_message(
                    chat_id=cid,
                    text="⚠️ Авто-анализ не удался. Нажмите *🔍 Анализ* вручную.",
                    parse_mode=ParseMode.MARKDOWN,
                    reply_markup=kb_main(),
                )
            except Exception:
                pass



async def _on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Global error handler to surface Conflict and shut down cleanly."""
    try:
        import telegram
        err = context.error
        if isinstance(err, telegram.error.Conflict):
            log.error("Telegram Conflict (another getUpdates active). Shutting down.")
            # Notify known chats
            for cid in list(_known_chat_ids):
                try:
                    await _send(cid, "⚠️ *Telegram Conflict 409*: другой экземпляр бота уже запущен. Остановите его и перезапустите текущий.", context.application)
                except Exception:
                    pass
            # Stop application
            try:
                await context.application.stop()
            except Exception:
                pass
            return
    except Exception:
        pass
    # default log
    log.exception("Unhandled error: %s", context.error)

def main() -> None:
    token = config.TELEGRAM_BOT_TOKEN
    if not token:
        raise RuntimeError(
            "Токен не задан.\nСоздайте файл .env:\n"
            "TELEGRAM_BOT_TOKEN=ваш_токен"
        )
    app = (
        Application.builder()
        .token(token)
        .post_init(_post_init)
        .build()
    )
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("menu",  cmd_start))
    app.add_handler(CommandHandler("positions", cmd_positions))
    app.add_handler(CommandHandler("why",   cmd_why))
    app.add_handler(CommandHandler("test",  cmd_test))
    app.add_handler(CallbackQueryHandler(btn, block=False))
    app.add_error_handler(_on_error)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))
    log.info("Bot started.")
    app.run_polling(drop_pending_updates=True, poll_interval=0.0, timeout=10)


if __name__ == "__main__":
    main()
