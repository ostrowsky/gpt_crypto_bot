from __future__ import annotations


def is_open_menu_text(text: str) -> bool:
    value = str(text or "").strip()
    lower = value.lower()
    return (
        value.startswith("📋")
        or (("открыть" in lower or "open" in lower) and ("меню" in lower or "menu" in lower))
        or lower in {"/menu", "menu"}
    )


def is_hide_menu_text(text: str) -> bool:
    value = str(text or "").strip()
    lower = value.lower()
    return (
        value.startswith("🙈")
        or (("скрыть" in lower or "hide" in lower) and ("меню" in lower or "menu" in lower))
    )
