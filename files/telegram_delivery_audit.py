from __future__ import annotations

import hashlib
import re
from typing import Any


_SYMBOL_RE = re.compile(r"(?<![A-Z0-9])([A-Z0-9]{2,20}USDT)(?![A-Z0-9])")
_TF_RE = re.compile(r"\[(15m|1h|4h|1d)\]")
_MARKDOWN_SYMBOL_RE = re.compile(r"\*([A-Z0-9]{2,20}USDT)\*")


def chat_id_hash(chat_id: int | str) -> str:
    raw = str(chat_id).encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()[:16]


def classify_message(text: str) -> dict[str, Any]:
    text = str(text or "")
    lower = text.lower()
    if "crypto trend bot" in lower or "кнопки меню" in lower or "меню" in lower and "портфель" in lower:
        kind = "menu"
    elif "сигнал покупки" in lower or "buy signal" in lower:
        kind = "buy"
    elif "сигнал продажи" in lower or "sell signal" in lower:
        kind = "sell"
    elif "watch only" in lower or "shadow re-entry" in lower:
        kind = "watch"
    elif "v2 shadow" in lower:
        kind = "v2_shadow"
    elif "единый портфель" in lower or "portfolio" in lower:
        kind = "portfolio"
    else:
        kind = "other"

    symbol = None
    markdown_match = _MARKDOWN_SYMBOL_RE.search(text)
    if markdown_match:
        symbol = markdown_match.group(1)
    else:
        symbol_match = _SYMBOL_RE.search(text)
        if symbol_match:
            symbol = symbol_match.group(1)

    tf = None
    tf_match = _TF_RE.search(text)
    if tf_match:
        tf = tf_match.group(1)

    preview = " ".join(text.replace("`", "").replace("*", "").split())
    if len(preview) > 180:
        preview = preview[:177] + "..."
    return {
        "message_kind": kind,
        "sym": symbol,
        "tf": tf,
        "text_preview": preview,
    }
