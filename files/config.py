from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# в”Ђв”Ђ Telegram в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")

# в”Ђв”Ђ Binance в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
BINANCE_REST: str = "https://api.binance.com"

# в”Ђв”Ђ Watchlist в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
WATCHLIST_FILE = Path("watchlist.json")

DEFAULT_WATCHLIST: list[str] = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT",
    "TRXUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "SHIBUSDT",
    "DOTUSDT", "LINKUSDT", "LTCUSDT", "BCHUSDT", "UNIUSDT",
    "TONUSDT", "NEARUSDT", "ICPUSDT", "AAVEUSDT", "HBARUSDT",
    "ARBUSDT", "OPUSDT", "POLUSDT", "SUIUSDT", "APTUSDT",
    "STRKUSDT", "ZROUSDT", "UMAUSDT", "ENAUSDT", "SEIUSDT",
    "METISUSDT", "MKRUSDT", "CRVUSDT", "SUSHIUSDT", "COMPUSDT",
    "YFIUSDT", "SNXUSDT", "LDOUSDT", "1INCHUSDT", "DYDXUSDT",
    "FETUSDT", "RNDRUSDT", "TAOUSDT", "INJUSDT", "ATOMUSDT",
    "ALGOUSDT", "XLMUSDT", "XTZUSDT", "EOSUSDT", "CELOUSDT",
    "ETCUSDT", "FILUSDT", "EGLDUSDT", "PAXGUSDT", "QNTUSDT",
    "RUNEUSDT", "CAKEUSDT", "GRTUSDT", "AXSUSDT", "SANDUSDT",
    "MANAUSDT", "CHZUSDT", "APEUSDT", "FLOKIUSDT", "WIFUSDT",
    "BONKUSDT", "ILVUSDT", "AUDIOUSDT", "JASMYUSDT", "ACHUSDT",
    "CFXUSDT", "ENSUSDT", "GMTUSDT", "ORDIUSDT", "WLDUSDT",
    "BLURUSDT", "LRCUSDT", "ZRXUSDT", "ZILUSDT", "KSMUSDT",
    "BATUSDT", "AMPUSDT", "BNTUSDT", "MDTUSDT", "GLMUSDT",
    "FLUXUSDT", "OXTUSDT", "BAKEUSDT", "PYRUSDT", "TRUUSDT",
    "ARUSDT", "COTIUSDT", "CELRUSDT", "QIUSDT", "SNTUSDT",
    "AXLUSDT", "TIAUSDT", "AEVOUSDT", "RENDERUSDT", "XAIUSDT",
    "C98USDT", "ACAUSDT", "LQTYUSDT", "DOGSUSDT", "MEMEUSDT",
]

def load_watchlist() -> list[str]:
    if WATCHLIST_FILE.exists():
        raw = json.loads(WATCHLIST_FILE.read_text())
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in raw:
            sym = str(item or "").strip().upper()
            if not sym:
                continue
            if sym == "MENU":
                continue
            if not sym.isalnum():
                continue
            if sym in seen:
                continue
            cleaned.append(sym)
            seen.add(sym)
        return cleaned
    return list(DEFAULT_WATCHLIST)


def save_watchlist(symbols: list[str]) -> None:
    WATCHLIST_FILE.write_text(json.dumps(symbols, indent=2))


# в”Ђв”Ђ Timeframes в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
TIMEFRAMES: list[str] = ["15m", "1h"]

# в”Ђв”Ђ Wide market scan в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
SCAN_TOP_N:   int       = 50
SCAN_QUOTE:   str       = "USDT"
SCAN_EXCLUDE: list[str] = [
    "UP", "DOWN", "BULL", "BEAR",
    "USDCUSDT", "BUSDUSDT", "TUSDUSDT", "DAIUSDT", "FDUSDUSDT",
]

# в”Ђв”Ђ Indicator parameters в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
EMA_FAST       = 20
EMA_SLOW       = 50
RSI_PERIOD     = 14
ADX_PERIOD     = 14
ATR_PERIOD     = 14
VOL_LOOKBACK   = 20
SLOPE_LOOKBACK = 5

# в”Ђв”Ђ Entry conditions в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
EMA_SLOPE_MIN  = 0.10
ADX_MIN        = 20.0
ADX_GROW_BARS  = 3   # РёСЃРїРѕР»СЊР·СѓРµС‚СЃСЏ С‚РѕР»СЊРєРѕ РІ РІС‹С…РѕРґРµ (ADX РѕСЃР»Р°Р±)
ADX_SMA_PERIOD = 10  # РїРµСЂРёРѕРґ SMA РґР»СЏ С„РёР»СЊС‚СЂР° ADX РЅР° РІС…РѕРґРµ
VOL_MULT       = 1.30
RSI_BUY_LO         = 45.0
RSI_BUY_HI         = 72.0   # СЃС‚Р°РЅРґР°СЂС‚РЅР°СЏ РІРµСЂС…РЅСЏСЏ РіСЂР°РЅРёС†Р° RSI
RSI_BUY_HI_STRONG  = 80.0   # СЂР°СЃС€РёСЂРµРЅРЅР°СЏ РіСЂР°РЅРёС†Р° РїСЂРё СЃРёР»СЊРЅРѕРј С‚СЂРµРЅРґРµ
STRONG_ADX_MIN     = 28.0   # ADX в‰Ґ СЌС‚РѕРіРѕ в†’ "СЃРёР»СЊРЅС‹Р№ С‚СЂРµРЅРґ"
STRONG_VOL_MIN     = 2.0    # volГ— в‰Ґ СЌС‚РѕРіРѕ в†’ "СЃРёР»СЊРЅС‹Р№ РѕР±СЉС‘Рј"
# РџСЂРё ADX в‰Ґ STRONG_ADX_MIN Р volГ— в‰Ґ STRONG_VOL_MIN в†’ RSI СЂР°Р·СЂРµС€С‘РЅ РґРѕ RSI_BUY_HI_STRONG

# в”Ђв”Ђ IMPULSE: РґРµС‚РµРєС‚РѕСЂ РЅР°С‡Р°Р»Р° С‚СЂРµРЅРґР° в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
# РЎСЂР°Р±Р°С‚С‹РІР°РµС‚ РІ СЃР°РјРѕРј РЅР°С‡Р°Р»Рµ РґРІРёР¶РµРЅРёСЏ вЂ” РґРѕ С‚РѕРіРѕ РєР°Рє ADX СѓСЃРїРµРІР°РµС‚ РІС‹СЂР°СЃС‚Рё.
# РљР»СЋС‡РµРІРѕРµ РѕС‚Р»РёС‡РёРµ: ADX РЅРµ С‚СЂРµР±СѓРµС‚СЃСЏ, РІРјРµСЃС‚Рѕ РЅРµРіРѕ вЂ” РѕР±СЉС‘Рј Рё СЃРєРѕСЂРѕСЃС‚СЊ С†РµРЅС‹.

IMPULSE_VOL_MIN:      float = 2.0   # vol_x РјРёРЅРёРјСѓРј вЂ” РЅСѓР¶РµРЅ СЂРµР°Р»СЊРЅС‹Р№ РѕР±СЉС‘Рј
IMPULSE_PRICE_SPEED:  float = 1.0   # % СЂРѕСЃС‚ С†РµРЅС‹ Р·Р° IMPULSE_SPEED_BARS Р±Р°СЂРѕРІ
IMPULSE_SPEED_BARS:   int   = 4     # РѕРєРЅРѕ РґР»СЏ РѕС†РµРЅРєРё СЃРєРѕСЂРѕСЃС‚Рё (4 Р±Р°СЂР° = 1С‡ РЅР° 15m)
IMPULSE_RANGE_MAX:    float = 5.0   # daily_range < 5% вЂ” РµС‰С‘ РЅРµ РїРѕР·РґРЅРѕ РІС…РѕРґРёС‚СЊ
IMPULSE_RSI_LO:       float = 50.0  # RSI РІС‹С€Рµ РЅРµР№С‚СЂР°Р»Рё
IMPULSE_RSI_HI:       float = 72.0  # RSI РµС‰С‘ РЅРµ РїРµСЂРµРіСЂРµС‚
IMPULSE_CROSS_BARS:   int   = 6     # РѕРєРЅРѕ РїРѕРёСЃРєР° РїРµСЂРµСЃРµС‡РµРЅРёСЏ EMA20>EMA50

# РЎРєР°РЅРёСЂРѕРІР°РЅРёРµ IMPULSE: РЅРµР·Р°РІРёСЃРёРјР°СЏ С„РѕРЅРѕРІР°СЏ Р·Р°РґР°С‡Р°
IMPULSE_SCAN_SEC:     int   = 900   # РєР°Р¶РґС‹Рµ 15 РјРёРЅСѓС‚ (= 1 Р±Р°СЂ РЅР° 15m)
IMPULSE_COOLDOWN_SEC: int   = 3600  # РЅРµ РїРѕРІС‚РѕСЂСЏС‚СЊ СЃРёРіРЅР°Р» РїРѕ РѕРґРЅРѕР№ РјРѕРЅРµС‚Рµ С‡Р°С‰Рµ 1С‡


# в”Ђв”Ђ Exit conditions в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
RSI_OVERBOUGHT = 85.0
ADX_DROP_RATIO = 0.75

# в”Ђв”Ђ Forward accuracy в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
FORWARD_BARS = [3, 5, 10]
MIN_ACCURACY = 60.0
MIN_SIGNALS  = 5  # РѕР±С‰РёР№ РјРёРЅРёРјСѓРј (РЅРµ РёСЃРїРѕР»СЊР·СѓРµС‚СЃСЏ РІ РґРЅРµРІРЅРѕРј)

# в”Ђв”Ђ Live monitoring в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
POLL_SEC      = 60
HISTORY_LIMIT = 300
LIVE_LIMIT    = 100

# в”Ђв”Ђ РќРѕРІС‹Рµ С„РёР»СЊС‚СЂС‹ РІС…РѕРґР° в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
# РњР°РєСЃ. СЂРѕСЃС‚ РѕС‚ РјРёРЅРёРјСѓРјР° РїРѕСЃР»РµРґРЅРёС… 96 Р±Р°СЂРѕРІ (24С‡ РЅР° 15m)
# Р•СЃР»Рё РјРѕРЅРµС‚Р° СѓР¶Рµ РІС‹СЂРѕСЃР»Р° Р±РѕР»СЊС€Рµ вЂ” С‚СЂРµРЅРґ СѓСЃС‚Р°Р», РІС…РѕРґ Р·Р°РїСЂРµС‰С‘РЅ
DAILY_RANGE_MAX: float = 7.0

# РњР°РєСЃРёРјСѓРј Р±Р°СЂРѕРІ РІ РїРѕР·РёС†РёРё вЂ” РµСЃР»Рё Р·Р° СЌС‚Рѕ РІСЂРµРјСЏ РЅРµС‚ РІС‹С…РѕРґР° РїРѕ СѓСЃР»РѕРІРёСЏРј,
# РІС‹С…РѕРґРёРј РїСЂРёРЅСѓРґРёС‚РµР»СЊРЅРѕ. РќР° 15m: 16 Р±Р°СЂРѕРІ = 4 С‡Р°СЃР°
MAX_HOLD_BARS: int = 16
TIME_EXIT_TREND_CONTINUATION_ENABLED: bool = True
TIME_EXIT_CONTINUE_CLOSE_ABOVE_EMA20: bool = True
TIME_EXIT_CONTINUE_SLOPE_MIN: float = 0.0
TIME_EXIT_CONTINUE_RSI_MIN: float = 50.0
TIME_EXIT_CONTINUE_MACD_HIST_MIN: float = 0.0

# РњРёРЅРёРјСѓРј РѕС†РµРЅС‘РЅРЅС‹С… СЃРёРіРЅР°Р»РѕРІ СЃРµРіРѕРґРЅСЏ РґР»СЏ РїРѕРґС‚РІРµСЂР¶РґРµРЅРёСЏ СЃС‚СЂР°С‚РµРіРёРё
# Р•СЃР»Рё РјРµРЅСЊС€Рµ вЂ” РјРѕРЅРµС‚Р° РЅРµ РїСЂРѕС…РѕРґРёС‚ РІ РјРѕРЅРёС‚РѕСЂРёРЅРі
TODAY_MIN_SIGNALS: int = 2
FORWARD_TEST_WINDOW_HOURS: int = 24  # СЃРєРѕР»СЊР·СЏС‰РµРµ РѕРєРЅРѕ С„РѕСЂРІР°СЂРґ-С‚РµСЃС‚Р° (РІРјРµСЃС‚Рѕ UTC-РїРѕР»РЅРѕС‡СЊ)

# РњРёРЅРёРјР°Р»СЊРЅР°СЏ С‚РѕС‡РЅРѕСЃС‚СЊ T+3 РґР»СЏ РїРѕРґС‚РІРµСЂР¶РґРµРЅРёСЏ (Р±С‹Р»Рѕ 50%, СЃС‚Р°Р»Рѕ СЃС‚СЂРѕР¶Рµ)
TODAY_T3_MIN: float = 60.0

# РњРёРЅРёРјР°Р»СЊРЅР°СЏ С‚РѕС‡РЅРѕСЃС‚СЊ T+10 вЂ” РµСЃР»Рё РЅРёР¶Рµ СЌС‚РѕРіРѕ, РјРѕРЅРµС‚Р° РѕРїР°СЃРЅР° РЅР° РґР»РёРЅРЅРѕРј РіРѕСЂРёР·РѕРЅС‚Рµ
TODAY_T10_MIN: float = 40.0

# РРЅС‚РµСЂРІР°Р» Р°РІС‚Рѕ-СЂРµР°РЅР°Р»РёР·Р° РІ СЃРµРєСѓРЅРґР°С… (0 = РІС‹РєР»СЋС‡РµРЅ)
# 7200 = РєР°Р¶РґС‹Рµ 2 С‡Р°СЃР° Р±РѕС‚ СЃР°Рј РїРµСЂРµСЃС‡РёС‚С‹РІР°РµС‚ СЃРїРёСЃРѕРє РјРѕРЅРµС‚
AUTO_REANALYZE_SEC: int = 900
AUTO_REANALYZE_TELEGRAM_REPORTS_ENABLED: bool = False
BOT_ENABLE_DATA_COLLECTOR: bool = False
BOT_STARTUP_AUTO_SCAN_ENABLED: bool = True
RL_WORKER_ENABLE_COLLECTOR: bool = True

# Research-only wide Binance universe collector. It does not trade, does not
# emit Telegram alerts, and does not write to production ML/critic datasets.
RESEARCH_UNIVERSE_SHADOW_ENABLED: bool = True
RESEARCH_UNIVERSE_SHADOW_MAX_SYMBOLS: int = 80
RESEARCH_UNIVERSE_SHADOW_TIMEFRAMES: tuple[str, ...] = ("15m",)
RESEARCH_UNIVERSE_SHADOW_MIN_QUOTE_VOLUME: float = 1_000_000.0
RESEARCH_UNIVERSE_SHADOW_BATCH_SIZE: int = 8
RESEARCH_UNIVERSE_SHADOW_INTERVAL_SEC: int = 15 * 60
RESEARCH_UNIVERSE_SHADOW_SYMBOL_TIMEOUT_SEC: int = 40
RESEARCH_UNIVERSE_SHADOW_SCORECARD_ENABLED: bool = True
RESEARCH_UNIVERSE_SHADOW_SCORECARD_TELEGRAM_ENABLED: bool = False
RESEARCH_UNIVERSE_SHADOW_SCORECARD_DAYS: int = 14
RESEARCH_UNIVERSE_SHADOW_SCORECARD_HORIZON: int = 5

ATR_TRAIL_K: float = 2.0   # РјРЅРѕР¶РёС‚РµР»СЊ ATR РґР»СЏ С‚СЂРµР№Р»РёРЅРі-СЃС‚РѕРїР°
MACDWARN_BARS: int = 3     # Р±Р°СЂРѕРІ РїРѕРґСЂСЏРґ MACD hist РїР°РґР°РµС‚ в†’ РїСЂРµРґСѓРїСЂРµР¶РґРµРЅРёРµ Рѕ СЂР°Р·РІРѕСЂРѕС‚Рµ

# в”Ђв”Ђ Рџ1: ATR-С‚СЂРµР№Р» Рё Р»РёРјРёС‚ Р±Р°СЂРѕРІ РїРѕ СЂРµР¶РёРјСѓ РІС…РѕРґР° в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
ATR_TRAIL_K_STRONG:    float = 2.5   # BUY strong_trend (ADX РІС‹СЃРѕРєРёР№ + РѕР±СЉС‘Рј)
ATR_TRAIL_K_RETEST:    float = 1.8   # RETEST (РѕС‚РєР°С‚ Рє EMA20 вЂ” СЂРёСЃРє РЅРёР¶Рµ)
ATR_TRAIL_K_BREAKOUT:  float = 1.5   # BREAKOUT (РїСЂРѕР±РѕР№ С„Р»СЌС‚Р° вЂ” Р±С‹СЃС‚СЂС‹Р№ РІС‹С…РѕРґ)
MAX_HOLD_BARS_RETEST:  int   = 10    # RETEST: 10 Р±Р°СЂРѕРІ Г— 15m = 2.5 С‡Р°СЃР°
MAX_HOLD_BARS_BREAKOUT:int   = 6     # BREAKOUT: 6 Р±Р°СЂРѕРІ Г— 15m = 1.5 С‡Р°СЃР°

# в”Ђв”Ђ Рџ5: Trend Day (BTC 1h EMA50) в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
# Р’ Р±С‹С‡РёР№ РґРµРЅСЊ СЂР°СЃС€РёСЂСЏРµРј РґРѕРїСѓСЃС‚РёРјС‹Рµ РїРѕСЂРѕРіРё
BULL_DAY_RANGE_MAX: float = 14.0  # DAILY_RANGE_MAX РїСЂРё Р±С‹С‡СЊРµРј РґРЅРµ
BULL_DAY_RSI_HI:    float = 75.0  # RSI_BUY_HI РїСЂРё Р±С‹С‡СЊРµРј РґРЅРµ

# в”Ђв”Ђ ADX SMA bypass (СѓР¶Рµ РёСЃРїРѕР»СЊР·РѕРІР°Р»СЃСЏ С‡РµСЂРµР· getattr) в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
ADX_SMA_BYPASS: float = 35.0  # ADX в‰Ґ СЌС‚РѕРіРѕ в†’ РїР»Р°С‚Рѕ СЃРёР»СЊРЅРѕРіРѕ С‚СЂРµРЅРґР°, bypass

# в”Ђв”Ђ Рџ3: Cooldown (СѓР¶Рµ РёСЃРїРѕР»СЊР·РѕРІР°Р»СЃСЏ С‡РµСЂРµР· getattr) в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
COOLDOWN_BARS: int = 8  # Р±Р°СЂРѕРІ С‚РёС€РёРЅС‹ РїРѕСЃР»Рµ РІС‹С…РѕРґР° (8 Г— 15m = 2 С‡Р°СЃР°)
AGENT_RESPECT_MAIN_EXIT_COOLDOWN: bool = True
AGENT_MAIN_EXIT_COOLDOWN_BARS: int = 8  # agent must not re-enter right after main bot SELL

# в”Ђв”Ђ RETEST: РѕС‚РєР°С‚ Рє EMA20 РІ СЃСѓС‰РµСЃС‚РІСѓСЋС‰РµРј С‚СЂРµРЅРґРµ в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
RETEST_LOOKBACK:    int   = 12    # Р±Р°СЂРѕРІ РЅР°Р·Р°Рґ вЂ” РїСЂРѕРІРµСЂСЏРµРј С‡С‚Рѕ С‚СЂРµРЅРґ Р±С‹Р»
RETEST_TOUCH_BARS:  int   = 5     # РѕРєРЅРѕ РїРѕРёСЃРєР° РєР°СЃР°РЅРёСЏ EMA20
RETEST_RSI_MAX:     float = 65.0  # RSI РЅР° СЂРµС‚РµСЃС‚Рµ РґРѕР»Р¶РµРЅ Р±С‹С‚СЊ РЅРёР¶Рµ
RETEST_VOL_MIN:     float = 0.8   # РѕР±СЉС‘Рј РґР»СЏ СЂРµС‚РµСЃС‚Р° РЅРµРѕР±СЏР·Р°С‚РµР»СЊРЅРѕ РІС‹СЃРѕРєРёР№

# в”Ђв”Ђ BREAKOUT: РїСЂРѕР±РѕР№ С„Р»СЌС‚Р° СЃ РѕР±СЉС‘РјРѕРј в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
BREAKOUT_FLAT_BARS:    int   = 8    # Р±Р°СЂРѕРІ С„Р»СЌС‚Р° РїРµСЂРµРґ РїСЂРѕР±РѕРµРј
BREAKOUT_FLAT_MAX_PCT: float = 2.0  # РјР°РєСЃ РґРёР°РїР°Р·РѕРЅ С„Р»СЌС‚Р° (%)
BREAKOUT_VOL_MIN:      float = 2.0  # vol_x РЅР° РїСЂРѕР±РѕРµ
BREAKOUT_RANGE_MAX:    float = 4.0  # daily_range вЂ” РґРІРёР¶РµРЅРёРµ С‚РѕР»СЊРєРѕ РЅР°С‡Р°Р»РѕСЃСЊ

# в”Ђв”Ђ IMPULSE: РґРµС‚РµРєС‚РѕСЂ РЅР°С‡Р°Р»Р° РёРјРїСѓР»СЊСЃР° (РґРѕ РїРѕРґС‚РІРµСЂР¶РґРµРЅРёСЏ ADX) в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
# РћС‚РєР°Р»РёР±СЂРѕРІР°РЅ РїРѕ СЂРµР°Р»СЊРЅС‹Рј РґР°РЅРЅС‹Рј 04.03.2026:
#   ETH 15:15 вЂ” r1=+2.37% r3=+3.50% RSI=78.8 volГ—2.63  в†ђ РїРѕР№РјР°Р» Р·Р° 1 Р±Р°СЂ РґРѕ BUY
#   SOL 15:15 вЂ” r1=+2.11% r3=+3.27% RSI=76.4 volГ—2.35
#   XRP 15:15 вЂ” r1=+2.05% r3=+2.69% RSI=79.5 volГ—3.77
#   XLMUSDT  вЂ” r1=+1.54% r3=+2.13% RSI=72.6 volГ—1.73
IMPULSE_R1_MIN:        float = 1.5   # РјРёРЅ СЂРѕСЃС‚ С‚РµРєСѓС‰РµРіРѕ Р±Р°СЂР° (%)
IMPULSE_R3_MIN:        float = 2.0   # РјРёРЅ СЂРѕСЃС‚ Р·Р° 3 Р±Р°СЂР° (%)
IMPULSE_VOL_MIN:       float = 1.5   # РјРёРЅ РѕР±СЉС‘Рј РєСЂР°С‚РЅС‹Р№ СЃСЂРµРґРЅРµРјСѓ
IMPULSE_BODY_MIN:      float = 0.5   # РјРёРЅ С‚РµР»Рѕ СЃРІРµС‡Рё (%) вЂ” СЂРµР°Р»СЊРЅРѕРµ РґРІРёР¶РµРЅРёРµ
IMPULSE_RSI_LO:        float = 45.0  # RSI РЅРёР¶РЅСЏСЏ РіСЂР°РЅРёС†Р°
IMPULSE_RSI_HI:        float = 80.0  # RSI РІРµСЂС…РЅСЏСЏ (80 вЂ” Р»РѕРІРёРј РёРјРїСѓР»СЊСЃ РїСЂРё СЂР°Р·РіРѕРЅРµ)
IMPULSE_COOLDOWN_BARS: int   = 8     # Р±Р°СЂРѕРІ РјРµР¶РґСѓ СЃРёРіРЅР°Р»Р°РјРё РѕРґРЅРѕР№ РјРѕРЅРµС‚С‹

# в”Ђв”Ђ TREND_SURGE: РґРµС‚РµРєС‚РѕСЂ РЅР°С‡Р°Р»Р° СѓСЃС‚РѕР№С‡РёРІРѕРіРѕ С‚СЂРµРЅРґР° в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
# Р›РѕРІРёС‚ РјРѕРјРµРЅС‚ РєРѕРіРґР° С‚СЂРµРЅРґ В«РІРєР»СЋС‡Р°РµС‚СЃСЏВ» вЂ” slope СѓСЃРєРѕСЂСЏРµС‚СЃСЏ + MACD СЂР°СЃС‚С‘С‚.
# РќРµ Р·Р°РІРёСЃРёС‚ РѕС‚ ADX Рё С„РѕСЂРІР°СЂРґ-С‚РµСЃС‚Р°. РљСѓР»РґР°СѓРЅ 5 С‡Р°СЃРѕРІ вЂ” РѕРґРёРЅ СЃРёРіРЅР°Р» РЅР° С‚СЂРµРЅРґ.
# РџСЂРёРјРµСЂС‹: JASMY 09.03 03:00 UTC (+8% Р·Р° 12С‡), BONK 09.03 12:00 UTC.
SURGE_SLOPE_MIN:     float = 0.15   # slope EMA20 (%) вЂ” С‚СЂРµРЅРґ РґРѕР»Р¶РµРЅ СѓСЃРєРѕСЂРёС‚СЊСЃСЏ
SURGE_VOL_MIN:       float = 1.5    # РѕР±СЉС‘Рј РІС‹С€Рµ СЃСЂРµРґРЅРµРіРѕ
SURGE_RSI_LO:        float = 50.0   # RSI РІ Р·РѕРЅРµ РёРјРїСѓР»СЊСЃР°
SURGE_RSI_HI:        float = 80.0   # РЅРµ РїРµСЂРµРіСЂРµС‚
SURGE_COOLDOWN_BARS: int   = 20     # 20 Г— 15m = 5 С‡Р°СЃРѕРІ РјРµР¶РґСѓ СЃРёРіРЅР°Р»Р°РјРё РѕРґРЅРѕР№ РјРѕРЅРµС‚С‹

# в”Ђв”Ђ ALIGNMENT: РїР»Р°РІРЅС‹Р№ Р±С‹С‡РёР№ С‚СЂРµРЅРґ Р±РµР· ADX в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
# Р›РѕРІРёС‚ РјРµРґР»РµРЅРЅС‹Рµ Р°Р»СЊС‚-С‚СЂРµРЅРґС‹ РіРґРµ ADX РЅРµ СѓСЃРїРµРІР°РµС‚ РїРѕРґС‚РІРµСЂРґРёС‚СЊ Р·Р° 28+ Р±Р°СЂРѕРІ,
# РЅРѕ СЃС‚СЂСѓРєС‚СѓСЂР° СѓСЃС‚РѕР№С‡РёРІРѕ Р±С‹С‡СЊСЏ (РїСЂРёРјРµСЂ: CHZ 08.03 09:00-18:00 = +8% Р·Р° 9 С‡Р°СЃРѕРІ).
ALIGNMENT_SLOPE_MIN:  float = 0.05  # РјСЏРіС‡Рµ С‡РµРј BUY (0.10) вЂ” С‚СЂРµРЅРґ РјРѕР¶РµС‚ Р±С‹С‚СЊ РїР»Р°РІРЅС‹Рј
ALIGNMENT_VOL_MIN:    float = 0.8   # РЅРµ РЅСѓР¶РµРЅ СЃРїР°Р№Рє, РґРѕСЃС‚Р°С‚РѕС‡РЅРѕ Р»СЋР±РѕР№ Р°РєС‚РёРІРЅРѕСЃС‚Рё
ALIGNMENT_RSI_LO:     float = 45.0  # RSI РІС‹С€Рµ РЅРµР№С‚СЂР°Р»Рё
ALIGNMENT_RSI_HI:     float = 82.0  # РјРµРґР»РµРЅРЅС‹Р№ С‚СЂРµРЅРґ вЂ” RSI СЂР°Р·РѕРіСЂРµРІР°РµС‚СЃСЏ РїРѕСЃС‚РµРїРµРЅРЅРѕ
ALIGNMENT_RANGE_MAX:  float = 18.0  # в†‘ СЃ 12% вЂ” РјРµРґР»РµРЅРЅС‹Рµ С‚СЂРµРЅРґС‹ РёРґСѓС‚ РґРѕР»РіРѕ (Р±С‹Р»Рѕ 12%)
ALIGNMENT_MACD_BARS:  int   = 3     # MACD hist > 0 РїРѕСЃР»РµРґРЅРёРµ N Р±Р°СЂРѕРІ РїРѕРґСЂСЏРґ вЂ” РЅРµ РѕРґРЅРѕСЂР°Р·РѕРІС‹Р№ РІСЃРїР»РµСЃРє

# в”Ђв”Ђ IMPULSE: РїРѕРґРЅСЏС‚Р°СЏ РІРµСЂС…РЅСЏСЏ РіСЂР°РЅРёС†Р° RSI в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
# в†‘ СЃ 80 РґРѕ 82 вЂ” Р»РѕРІРёС‚ СЂР°Р·РіРѕРЅСЏСЋС‰РёРµСЃСЏ Р°Р»СЊС‚РєРѕРёРЅС‹ С‚РёРїР° XAI (+34%) СЂР°РЅСЊС€Рµ
IMPULSE_RSI_HI:       float = 82.0  # (РїРµСЂРµРѕРїСЂРµРґРµР»СЏРµС‚ РІС‹С€Рµ)

# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
# РќРћР’Р«Р• РџРђР РђРњР•РўР Р« v2: РќР°С‡Р°Р»Рѕ/РѕРєРѕРЅС‡Р°РЅРёРµ С‚СЂРµРЅРґР° вЂ” СЂР°СЃС€РёСЂРµРЅРЅР°СЏ СЃРёРіРЅР°Р»РёР·Р°С†РёСЏ
# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ

# в”Ђв”Ђ A. РЈСЃРєРѕСЂРµРЅРёРµ РЅР°РєР»РѕРЅР° EMA (slope acceleration) в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
# Р›РѕРІРёС‚ РјРѕРјРµРЅС‚ РєРѕРіРґР° EMA20 РЅР°С‡РёРЅР°РµС‚ СЂР°Р·РіРѕРЅСЏС‚СЊСЃСЏ вЂ” РѕРїРµСЂРµР¶Р°РµС‚ BUY РЅР° 1-3 Р±Р°СЂР°.
# slope[i] - slope[i-3] > SLOPE_ACCEL_MIN в†’ С‚СЂРµРЅРґ РЅР°Р±РёСЂР°РµС‚ СЃРёР»Сѓ РїСЂСЏРјРѕ СЃРµР№С‡Р°СЃ.
SLOPE_ACCEL_MIN:   float = 0.05   # СѓСЃРєРѕСЂРµРЅРёРµ slope (%) Р·Р° 3 Р±Р°СЂР°
SLOPE_ACCEL_BARS:  int   = 3      # РѕРєРЅРѕ РѕС†РµРЅРєРё СѓСЃРєРѕСЂРµРЅРёСЏ

# в”Ђв”Ђ B. Squeeze Breakout (РїСЂРѕР±РѕР№ СЃР¶Р°С‚РёСЏ ATR) в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
# ATR < 50% СЃРІРѕРµР№ N-Р±Р°СЂРѕРІРѕР№ СЃСЂРµРґРЅРµР№ = СЃР¶Р°С‚РёРµ (РЅР°РєРѕРїР»РµРЅРёРµ РїРµСЂРµРґ РґРІРёР¶РµРЅРёРµРј).
# ATR РІС‹СЂРѕСЃ РІ 1.8Г— РѕС‚ РјРёРЅРёРјСѓРјР° СЃР¶Р°С‚РёСЏ = РїСЂРѕР±РѕР№.
# РРґРµСЏ: Р±РѕРєРѕРІРёРє СЃР¶РёРјР°РµС‚ РїСЂСѓР¶РёРЅСѓ вЂ” РІС‹С…РѕРґ РёР· Р±РѕРєРѕРІРёРєР° РІР·СЂС‹РІРЅРѕР№.
SQUEEZE_LOOKBACK:       int   = 20    # Р±Р°СЂРѕРІ РґР»СЏ СЂР°СЃС‡С‘С‚Р° СЃСЂРµРґРЅРµР№ ATR
ATR_SQUEEZE_RATIO:      float = 0.5   # ATR < 50% РѕС‚ SMA(ATR,20) = СЃР¶Р°С‚РёРµ
ATR_EXPANSION_MULT:     float = 1.8   # ATR РІС‹СЂРѕСЃ РІ 1.8Г— РѕС‚ РґРЅР° СЃР¶Р°С‚РёСЏ = РїСЂРѕР±РѕР№
SQUEEZE_MIN_BARS:       int   = 5     # РјРёРЅРёРјСѓРј Р±Р°СЂРѕРІ РІ СЃР¶Р°С‚РёРё РїРµСЂРµРґ РїСЂРѕР±РѕРµРј

# в”Ђв”Ђ C. RSI Р”РёРІРµСЂРіРµРЅС†РёСЏ (СЃРёРіРЅР°Р» РѕРєРѕРЅС‡Р°РЅРёСЏ С‚СЂРµРЅРґР°) в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
# Р¦РµРЅР° РґРµР»Р°РµС‚ РЅРѕРІС‹Р№ maximum в†’ RSI РЅРµ РїРѕРґС‚РІРµСЂР¶РґР°РµС‚ в†’ СЃРєСЂС‹С‚РѕРµ РѕСЃР»Р°Р±Р»РµРЅРёРµ.
# РџСЂРё РѕР±РЅР°СЂСѓР¶РµРЅРёРё: СѓР¶РµСЃС‚РѕС‡РёС‚СЊ СЃС‚РѕРї (ATR_K * RSI_DIV_TRAIL_MULT), РЅРµ РѕС‚РєСЂС‹РІР°С‚СЊ BUY.
RSI_DIV_LOOKBACK:       int   = 10    # Р±Р°СЂРѕРІ РЅР°Р·Р°Рґ РґР»СЏ РїРѕРёСЃРєР° РїСЂРµРґС‹РґСѓС‰РµРіРѕ РјР°РєСЃРёРјСѓРјР°
RSI_DIV_PRICE_MARGIN:   float = 0.001 # С†РµРЅР° РґРѕР»Р¶РЅР° Р±С‹С‚СЊ РІС‹С€Рµ РЅР° >0.1% (С„РёР»СЊС‚СЂ С€СѓРјР°)
RSI_DIV_TRAIL_MULT:     float = 0.6   # РјРЅРѕР¶РёС‚РµР»СЊ ATR РїСЂРё РґРёРІРµСЂРіРµРЅС†РёРё (2.0 в†’ 1.2)

# в”Ђв”Ђ D. Volume Exhaustion (РёСЃС‚РѕС‰РµРЅРёРµ РѕР±СЉС‘РјР° вЂ” РєРѕРЅРµС† С‚СЂРµРЅРґР°) в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
# Р¦РµРЅР° СЂР°СЃС‚С‘С‚ N Р±Р°СЂРѕРІ РїРѕРґСЂСЏРґ, РЅРѕ РѕР±СЉС‘Рј РєР°Р¶РґС‹Р№ Р±Р°СЂ РЅРёР¶Рµ РїСЂРµРґС‹РґСѓС‰РµРіРѕ.
# РЎРёР»СЊРЅС‹Р№ СЃРёРіРЅР°Р» СЂР°Р·РІРѕСЂРѕС‚Р° вЂ” РїРѕРєСѓРїР°С‚РµР»Рё Р·Р°РєР°РЅС‡РёРІР°СЋС‚СЃСЏ.
VOL_EXHAUST_BARS:       int   = 5     # Р±Р°СЂРѕРІ СѓР±С‹РІР°СЋС‰РµРіРѕ РѕР±СЉС‘РјР° РїСЂРё СЂРѕСЃС‚Рµ С†РµРЅС‹
VOL_EXHAUST_PRICE_MIN:  float = 0.5   # РјРёРЅРёРјР°Р»СЊРЅС‹Р№ СЂРѕСЃС‚ С†РµРЅС‹ (%) Р·Р° СЌС‚Рё N Р±Р°СЂРѕРІ

# в”Ђв”Ђ E. EMA Fan Collapse (СЃС…Р»РѕРїС‹РІР°РЅРёРµ РІРµРµСЂР° вЂ” РєРѕРЅРµС† С‚СЂРµРЅРґР°) в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
# Р’ С‚СЂРµРЅРґРµ: EMA20 >> EMA50 >> EMA200, СЂР°СЃСЃС‚РѕСЏРЅРёСЏ СЂР°СЃС‚СѓС‚.
# Р Р°Р·РІРѕСЂРѕС‚: spread EMA20-EMA50 СѓРјРµРЅСЊС€РёР»СЃСЏ РЅР° SPREAD_DECAY РѕС‚ РјР°РєСЃРёРјСѓРјР°.
EMA_FAN_LOOKBACK:       int   = 8     # Р±Р°СЂРѕРІ РЅР°Р·Р°Рґ РґР»СЏ РїРѕРёСЃРєР° РјР°РєСЃРёРјСѓРјР° spread
EMA_FAN_DECAY_THRESHOLD: float = 0.30 # spread СѓРїР°Р» РЅР° 30% РѕС‚ РјР°РєСЃРёРјСѓРјР° в†’ РїСЂРµРґСѓРїСЂРµР¶РґРµРЅРёРµ

# в”Ђв”Ђ F. Market Regime (СЂРµР¶РёРј СЂС‹РЅРєР°) в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
# РЇРІРЅС‹Рµ СЂРµР¶РёРјС‹ СЂС‹РЅРєР° РјРµРЅСЏСЋС‚ РїРѕСЂРѕРіРё РґР»СЏ РІСЃРµС… СЃРёРіРЅР°Р»РѕРІ.
# BULL_TREND:    BTC > EMA50 + ADX > 25 в†’ РјСЏРіС‡Рµ RSI, range; СЃС‚СЂРѕР¶Рµ vol
# CONSOLIDATION: ADX < 20 в†’ СЃС‚СЂРѕР¶Рµ РІСЃС‘, Р¶РґС‘Рј РїСЂРѕР±РѕСЏ
# RECOVERY:      BTC РїСЂРѕР±РёРІР°РµС‚ EMA50 СЃРЅРёР·Сѓ в†’ Р°РіСЂРµСЃСЃРёРІРЅС‹Р№ РІС…РѕРґ
# BEAR_TREND:    BTC < EMA50 + ADX > 25 в†’ С‚РѕР»СЊРєРѕ СЂРµС‚РµСЃС‚С‹, Р·Р°РїСЂРµС‚ РЅРѕРІС‹С… BUY

# РџР°СЂР°РјРµС‚СЂС‹ РїРѕ СЂРµР¶РёРјР°Рј: {СЂРµР¶РёРј: {РїР°СЂР°РјРµС‚СЂ: Р·РЅР°С‡РµРЅРёРµ}}
REGIME_PARAMS: dict = {
    "bull_trend": {
        "rsi_hi":       75.0,
        "vol_mult":     1.1,
        "range_max":    10.0,
        "adx_min":      18.0,
        "slope_min":    0.08,
    },
    "consolidation": {
        "rsi_hi":       65.0,
        "vol_mult":     1.5,
        "range_max":    5.0,
        "adx_min":      22.0,
        "slope_min":    0.12,
    },
    "recovery": {
        "rsi_hi":       70.0,
        "vol_mult":     1.2,
        "range_max":    8.0,
        "adx_min":      18.0,
        "slope_min":    0.08,
    },
    "bear_trend": {
        "rsi_hi":       60.0,
        "vol_mult":     2.0,
        "range_max":    4.0,
        "adx_min":      25.0,
        "slope_min":    0.15,
    },
    "neutral": {
        # Р‘Р°Р·РѕРІС‹Рµ Р·РЅР°С‡РµРЅРёСЏ вЂ” РЅРµ РїРµСЂРµР·Р°РїРёСЃС‹РІР°РµРј config
        "rsi_hi":       None,
        "vol_mult":     None,
        "range_max":    None,
        "adx_min":      None,
        "slope_min":    None,
    },
}

# РџРѕСЂРѕРіРё РґР»СЏ РѕРїСЂРµРґРµР»РµРЅРёСЏ СЂРµР¶РёРјР° РїРѕ BTC 1h
REGIME_BTC_ADX_TREND:   float = 22.0   # ADX >= СЌС‚РѕРіРѕ в†’ С‚СЂРµРЅРґ (bull РёР»Рё bear)
REGIME_BTC_ADX_FLAT:    float = 18.0   # ADX < СЌС‚РѕРіРѕ в†’ РєРѕРЅСЃРѕР»РёРґР°С†РёСЏ
REGIME_BTC_RECOVERY_SLOPE: float = 0.05  # slope EMA50 РїСЂРё РїСЂРѕР±РѕРµ СЃРЅРёР·Сѓ РІРІРµСЂС…

# в”Ђв”Ђ G. Dynamic Range Max (Р°РґР°РїС‚РёРІРЅС‹Р№ РїРѕСЂРѕРі РїРѕ РІРѕР»Р°С‚РёР»СЊРЅРѕСЃС‚Рё РјРѕРЅРµС‚С‹) в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
# Р’РјРµСЃС‚Рѕ С„РёРєСЃРёСЂРѕРІР°РЅРЅРѕРіРѕ 7% вЂ” РїРѕСЂРѕРі РїСЂРѕРїРѕСЂС†РёРѕРЅР°Р»РµРЅ РёСЃС‚РѕСЂРёС‡РµСЃРєРѕР№ РІРѕР»Р°С‚РёР»СЊРЅРѕСЃС‚Рё РјРѕРЅРµС‚С‹.
# РњРѕРЅРµС‚С‹ С‚РёРїР° XAI (РґРЅРµРІРЅРѕР№ РґРёР°РїР°Р·РѕРЅ 15%+) РїРѕР»СѓС‡Р°СЋС‚ Р±РѕР»РµРµ С€РёСЂРѕРєРёР№ РїРѕСЂРѕРі.
# РњРѕРЅРµС‚С‹ С‚РёРїР° BTC (РґРёР°РїР°Р·РѕРЅ 3%) вЂ” Р±РѕР»РµРµ СѓР·РєРёР№.
DYNAMIC_RANGE_ENABLED:   bool  = True
DYNAMIC_RANGE_REF_PCT:   float = 5.0   # СЌС‚Р°Р»РѕРЅРЅС‹Р№ РґРЅРµРІРЅРѕР№ РґРёР°РїР°Р·РѕРЅ (РЅРѕСЂРјРёСЂРѕРІРєР°)
DYNAMIC_RANGE_HIST_BARS: int   = 96 * 14  # 14 РґРЅРµР№ РЅР° 15m РґР»СЏ СЂР°СЃС‡С‘С‚Р° avg_daily_range
DYNAMIC_RANGE_MIN:       float = 3.0   # РЅРёР¶РЅРёР№ РїСЂРµРґРµР» (Р·Р°С‰РёС‚Р° РѕС‚ СЃР»РёС€РєРѕРј СѓР·РєРѕРіРѕ РїРѕСЂРѕРіР°)
DYNAMIC_RANGE_MAX_CAP:   float = 25.0  # РІРµСЂС…РЅРёР№ РїСЂРµРґРµР» (Р·Р°С‰РёС‚Р° РѕС‚ Р±РµСЃРєРѕРЅРµС‡РЅРѕРіРѕ РїРѕСЂРѕРіР°)

# в”Ђв”Ђ Runtime overrides (СѓСЃС‚Р°РЅР°РІР»РёРІР°СЋС‚СЃСЏ РґРёРЅР°РјРёС‡РµСЃРєРё РІ market_scan / strategy) в”Ђв”Ђ
_current_regime:         str   = "neutral"
_regime_params_active:   dict  = {}     # Р°РєС‚РёРІРЅС‹Рµ РїР°СЂР°РјРµС‚СЂС‹ С‚РµРєСѓС‰РµРіРѕ СЂРµР¶РёРјР°

# в”Ђв”Ђ H. EMA Cross вЂ” СЂР°РЅРЅРёР№ СЃРёРіРЅР°Р» РїСЂРѕР±РѕСЏ EMA20 СЃРЅРёР·Сѓ РІРІРµСЂС… в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
# Р›РѕРІРёС‚ РјРѕРјРµРЅС‚ РєРѕРіРґР° С†РµРЅР° РїСЂРѕР±РёРІР°РµС‚ EMA20 СЃ РѕР±СЉС‘РјРѕРј Р”Рћ РїРѕРґС‚РІРµСЂР¶РґРµРЅРёСЏ slope/ADX.
# РўРёРїРёС‡РЅС‹Р№ РІС‹РёРіСЂС‹С€: 3-5 Р±Р°СЂРѕРІ (45-75 РјРёРЅСѓС‚) СЂР°РЅСЊС€Рµ СЃС‚Р°РЅРґР°СЂС‚РЅРѕРіРѕ СЃРёРіРЅР°Р»Р°.
#
# РџР°С‚С‚РµСЂРЅ: close[i-1] < ema20[i-1]  AND  close[i] >= ema20[i]  (РїСЂРѕР±РѕР№)
#          vol_x[i] >= CROSS_VOL_MIN                             (РѕР±СЉС‘Рј РїРѕРґС‚РІРµСЂР¶РґР°РµС‚)
#          ema50_slope >= CROSS_EMA50_SLOPE_MIN                  (EMA50 РЅРµ РїР°РґР°РµС‚)
#          RSI РІ РґРёР°РїР°Р·РѕРЅРµ [CROSS_RSI_LO, CROSS_RSI_HI]         (РЅРµ РїРµСЂРµРєСѓРїР»РµРЅ/РїРµСЂРµРїСЂРѕРґР°РЅ)
#          daily_range_pct <= CROSS_RANGE_MAX                    (РЅРµС‚ СѓР¶Рµ СЂР°Р·РѕРіРЅР°РЅРЅРѕРіРѕ С…РѕРґР°)
#          close > ema200 (РІС‹С€Рµ РґРѕР»РіРѕСЃСЂРѕС‡РЅРѕР№ РїРѕРґРґРµСЂР¶РєРё)          (РѕРїС†РёРѕРЅР°Р»СЊРЅРѕ)

CROSS_VOL_MIN:       float = 1.2    # РјРёРЅ РѕР±СЉС‘Рј РЅР° Р±Р°СЂРµ РїСЂРѕР±РѕСЏ (РЅРёР¶Рµ СЃС‚Р°РЅРґР°СЂС‚РЅРѕРіРѕ 1.3)
CROSS_EMA50_SLOPE_MIN: float = -0.40 # EMA50 РЅРµ РґРѕР»Р¶РЅР° РїР°РґР°С‚СЊ СЃРёР»СЊРЅРµРµ СЌС‚РѕРіРѕ % (Р·Р° 3 Р±Р°СЂР°)
CROSS_RSI_LO:        float = 38.0   # РЅРёР¶РЅСЏСЏ RSI (С€РёСЂРµ СЃС‚Р°РЅРґР°СЂС‚РЅРѕРіРѕ 45)
CROSS_RSI_HI:        float = 72.0   # РІРµСЂС…РЅСЏСЏ RSI (РєР°Рє СЃС‚Р°РЅРґР°СЂС‚, РІС‹С€Рµ = СѓР¶Рµ СЂР°Р·РѕРіРЅР°РЅ)
CROSS_RANGE_MAX:     float = 6.0    # РјР°РєСЃ РґРЅРµРІРЅРѕР№ РґРёР°РїР°Р·РѕРЅ (С„РёР»СЊС‚СЂ СѓР¶Рµ СЂР°Р·РѕРіРЅР°РЅРЅС‹С…)
CROSS_LOOKBACK:      int   = 3      # Р±Р°СЂРѕРІ РЅР°Р·Р°Рґ РґР»СЏ РїСЂРѕРІРµСЂРєРё С‡С‚Рѕ Р±С‹Р»Рѕ РЅРёР¶Рµ EMA20
CROSS_MACD_FILTER:   bool  = True   # С‚СЂРµР±РѕРІР°С‚СЊ MACD hist >= 0 РЅР° Р±Р°СЂРµ РїСЂРѕР±РѕСЏ
CROSS_COOLDOWN_BARS: int   = 6      # Р±Р°СЂРѕРІ РјРµР¶РґСѓ EMA_CROSS СЃРёРіРЅР°Р»Р°РјРё РѕРґРЅРѕР№ РјРѕРЅРµС‚С‹
CROSS_CONFIRM_BARS:  int   = 2      # РјР°РєСЃ Р±Р°СЂРѕРІ СЃ РјРѕРјРµРЅС‚Р° РїСЂРѕР±РѕСЏ (РЅРµ СЃР»Р°С‚СЊ СЃС‚Р°СЂРѕРµ)

# в”Ђв”Ђ Runtime: EMA_CROSS в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
_last_cross_ts:      dict  = {}     # sym в†’ timestamp РїРѕСЃР»РµРґРЅРµРіРѕ CROSS-СЃРёРіРЅР°Р»Р°


# в”Ђв”Ђ Portfolio / entry quality controls в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
MAX_OPEN_POSITIONS: int = 10
UNIFIED_PORTFOLIO_ENABLED: bool = True
UNIFIED_PORTFOLIO_MAX_POSITIONS: int = 10
UNIFIED_PORTFOLIO_AGENT_STATUS_MAX_AGE_SEC: int = 300
MAX_NEW_ENTRIES_PER_CYCLE: int = 1   # limit new entries per polling iteration
ENTRY_SCORE_PCTL: float = 80.0       # dynamic threshold percentile among candidates
ALIGNMENT_BUY_ENABLED: bool = False  # alignment is context/exit warning, not a buy trigger
MAX_POSITIONS_PER_GROUP: int = 2
OPEN_SIGNAL_CLUSTER_CAP_ENABLED: bool = True
OPEN_SIGNAL_CLUSTER_CAP_15M_SHORT_BOUNCE_MODES: tuple[str, ...] = ("breakout", "retest")
OPEN_SIGNAL_CLUSTER_CAP_15M_IMPULSE_MODES: tuple[str, ...] = ("impulse_speed",)
OPEN_SIGNAL_CLUSTER_CAP_15M_MOMENTUM_MODES: tuple[str, ...] = ("trend", "strong_trend", "impulse")
OPEN_SIGNAL_CLUSTER_CAP_15M_ALIGNMENT_MODES: tuple[str, ...] = ("alignment",)
OPEN_SIGNAL_CLUSTER_CAP_1H_RETEST_MODES: tuple[str, ...] = ("retest",)
OPEN_SIGNAL_CLUSTER_CAP_1H_MOMENTUM_MODES: tuple[str, ...] = ("trend", "strong_trend", "impulse_speed", "impulse")
OPEN_SIGNAL_CLUSTER_CAP_1H_ALIGNMENT_MODES: tuple[str, ...] = ("alignment",)
OPEN_SIGNAL_CLUSTER_CAP_15M_SHORT_BOUNCE_MAX: int = 2
OPEN_SIGNAL_CLUSTER_CAP_15M_IMPULSE_MAX: int = 2
OPEN_SIGNAL_CLUSTER_CAP_15M_MOMENTUM_MAX: int = 2
OPEN_SIGNAL_CLUSTER_CAP_15M_ALIGNMENT_MAX: int = 2
OPEN_SIGNAL_CLUSTER_CAP_1H_RETEST_MAX: int = 2
OPEN_SIGNAL_CLUSTER_CAP_1H_MOMENTUM_MAX: int = 2
OPEN_SIGNAL_CLUSTER_CAP_1H_ALIGNMENT_MAX: int = 2
OPEN_SIGNAL_CLUSTER_CAP_WATCH_ALERTS_ENABLED: bool = True
OPEN_SIGNAL_CLUSTER_CAP_WATCH_ALERT_MIN_SCORE: float = 80.0
OPEN_SIGNAL_CLUSTER_CAP_WATCH_ALERT_BUCKETS: tuple[str, ...] = ("15m_impulse", "1h_momentum", "1h_retest")

# Agent: use full portfolio capacity. Top-mover leader_score decides the top 10;
# do not let the coarse "momentum" mode cluster collapse the portfolio to 2.
AGENT_MAX_POSITIONS: int = 10
AGENT_ALLOWED_MODES: tuple[str, ...] = ("trend", "strong_trend", "impulse_speed", "4h_leader_watch")
AGENT_ALLOWED_TIMEFRAMES: tuple[str, ...] = ("15m", "1h")
AGENT_MIN_DAY_CHANGE_PCT: float = 1.25
AGENT_MIN_FORECAST_PROXY_PCT: float = 0.35
AGENT_MIN_LEADER_SCORE: float = 12.0
AGENT_MIN_ADX: float = 18.0
AGENT_TREND_MIN_ADX: float = 35.0
AGENT_MIN_VOL_X: float = 1.0
AGENT_MAX_RSI: float = 72.5
AGENT_MAX_DAILY_RANGE_PCT: float = 14.0
AGENT_MIN_TODAY_SIGNALS: int = 2
AGENT_MIN_BEST_ACCURACY: float = 55.0
AGENT_REQUIRE_DISTINCT_MODE_CLUSTERS: bool = False
AGENT_MAX_POSITIONS_PER_MODE_CLUSTER: int = 10
AGENT_MAX_POSITIONS_PER_GROUP: int = 2
AGENT_REPLACEMENT_ENABLED: bool = True
AGENT_REPLACEMENT_MIN_LEADER_DELTA: float = 0.0
AGENT_MAX_REPLACEMENTS_PER_CYCLE: int = 10
# Replay-confirmed replacement policy candidate. Default OFF: live behavior is
# unchanged until shadow logs prove it does not block rare winners.
AGENT_REPLACEMENT_BLOCK_NON_LOSING_ENABLED: bool = False
AGENT_REPLACEMENT_BLOCK_NON_LOSING_SHADOW: bool = True

# 4h context is not an entry trigger. It only adjusts ranking/leader score for
# valid 15m/1h candidates so higher-timeframe recovery can be noticed without
# reintroducing noisy standalone 4h BUY signals.
FOUR_H_CONTEXT_SCORE_ENABLED: bool = True
FOUR_H_CONTEXT_SCORE_WEIGHT: float = 1.0
FOUR_H_CONTEXT_LEADER_WEIGHT: float = 0.8
FOUR_H_CONTEXT_MAX_BONUS: float = 8.0
FOUR_H_CONTEXT_MAX_PENALTY: float = -6.0

# 4h leader watch: entry trigger for coins that already lead on 4h but were
# missed by normal 15m/1h patterns because generic intraday gates are too tight.
# It still requires fresh 15m/1h confirmation; 4h context alone is not enough.
AGENT_4H_LEADER_WATCH_ENABLED: bool = True
AGENT_4H_LEADER_MIN_CONTEXT_SCORE: float = 7.0
AGENT_4H_LEADER_MIN_TODAY_CHANGE_PCT: float = 4.0
AGENT_4H_LEADER_MAX_DAILY_RANGE_PCT: float = 35.0
AGENT_4H_LEADER_MIN_ADX: float = 30.0
AGENT_4H_LEADER_MIN_SLOPE: float = 0.35
AGENT_4H_LEADER_MIN_RSI: float = 50.0
AGENT_4H_LEADER_MAX_RSI: float = 78.0
AGENT_4H_LEADER_MIN_VOL_X: float = 0.35
AGENT_4H_LEADER_STRENGTH_GATE_ENABLED: bool = True
AGENT_4H_LEADER_STRENGTH_MIN_VOL_X: float = 3.0
AGENT_4H_LEADER_STRENGTH_MIN_TODAY_CHANGE_PCT: float = 10.0
AGENT_4H_LEADER_RECLAIM_MAX_PRICE_EDGE_PCT: float = 8.0
AGENT_4H_LEADER_PULLBACK_MAX_PRICE_EDGE_PCT: float = 3.5
AGENT_4H_LEADER_MIN_MACD_HIST: float = 0.0
AGENT_4H_LEADER_BONUS: float = 10.0
AGENT_4H_LEADER_TRAIL_K: float = 2.8
AGENT_4H_LEADER_MAX_HOLD_BARS_15M: int = 48
AGENT_4H_LEADER_MAX_HOLD_BARS_1H: int = 16
AGENT_4H_LEADER_BYPASS_SYMBOL_COOLDOWN: bool = True
AGENT_4H_LEADER_COOLDOWN_MIN_LEADER_SCORE: float = 55.0

# Main bot anti-chase gate: keep generic signal logic, but block extreme late
# top-gainer chases that are already far beyond a normal intraday leader move.
TOP_GAINER_CHASE_GUARD_ENABLED: bool = True
TOP_GAINER_CHASE_GUARD_MODES: tuple[str, ...] = ("trend", "strong_trend", "impulse_speed", "impulse")
TOP_GAINER_CHASE_GUARD_TIMEFRAMES: tuple[str, ...] = ("15m", "1h")
TOP_GAINER_CHASE_GUARD_MAX_RSI: float = 76.0
TOP_GAINER_CHASE_GUARD_RSI_RANGE_MIN_PCT: float = 8.0
TOP_GAINER_CHASE_GUARD_MAX_DAILY_RANGE_PCT: float = 25.0
TOP_GAINER_CHASE_GUARD_ALERTS_ENABLED: bool = True
TOP_GAINER_CHASE_GUARD_ALERT_MIN_CANDIDATE_SCORE: float = 100.0
TOP_GAINER_CHASE_GUARD_ALERT_MIN_LIVE_SCORE: float = 34.0
TOP_GAINER_CHASE_GUARD_LEARNING_LABEL_ENABLED: bool = True
TOP_GAINER_OBJECTIVE_GATE_ENABLED: bool = False
TOP_GAINER_OBJECTIVE_GATE_MODES: tuple[str, ...] = ("breakout", "retest", "trend", "strong_trend", "impulse_speed", "impulse")
TOP_GAINER_OBJECTIVE_MIN_INTRADAY_CHANGE_PCT: float = 0.35
TOP_GAINER_OBJECTIVE_MIN_DAILY_RANGE_PCT: float = 0.90
TOP_GAINER_OBJECTIVE_MIN_VOL_X: float = 1.20
TOP_GAINER_OBJECTIVE_MIN_ADX: float = 20.0
TOP_GAINER_OBJECTIVE_RETEST_MIN_INTRADAY_CHANGE_PCT: float = 0.75
TOP_GAINER_OBJECTIVE_MOMENTUM_MIN_INTRADAY_CHANGE_PCT: float = 1.00
TOP_GAINER_OBJECTIVE_STRONG_SCORE_BYPASS: float = 115.0
TOP_GAINER_OBJECTIVE_STRONG_ADX_BYPASS: float = 32.0
TOP_GAINER_OBJECTIVE_ALLOW_CONFIRMED_LEADER: bool = True
TOP_GAINER_SCORE_GATE_ENABLED: bool = True
TOP_GAINER_SCORE_GATE_MODES: tuple[str, ...] = ("breakout", "retest", "trend", "strong_trend", "impulse_speed", "impulse")
TOP_GAINER_SCORE_GATE_MIN_SCORE: float = 34.0
TOP_GAINER_SCORE_GATE_MODE_MIN_SCORE: dict[str, float] = {"impulse": 30.0}

# Watch-only alerts for strong candidates blocked by quality gates. These do not
# open positions; they explain "why no signal" without weakening production BUYs.
TOP_GAINER_WATCH_ALERTS_ENABLED: bool = True
TOP_GAINER_WATCH_ALERT_MODES: tuple[str, ...] = ("impulse_speed",)
TOP_GAINER_WATCH_ALERT_MIN_SCORE: float = 30.0
TOP_GAINER_SCORE_GATE_STRONG_ALERTS_ENABLED: bool = True
TOP_GAINER_SCORE_GATE_STRONG_ALERT_MIN_CANDIDATE_SCORE: float = 100.0
TOP_GAINER_SCORE_GATE_STRONG_ALERT_MIN_LIVE_SCORE: float = 28.0
TOP_GAINER_SCORE_GATE_STRONG_ALERT_MAX_DEFICIT: float = 6.0
TOP_GAINER_SCORE_GATE_LEARNING_LABEL_ENABLED: bool = True

# Shadow-only early top-mover scout. It never opens positions and never emits a
# live BUY; it writes hypothetical entries so the signal-quality evaluator can
# measure whether this scout would reduce missed early trends.
EARLY_TOP_MOVER_SCOUT_SHADOW_ENABLED: bool = True
EARLY_TOP_MOVER_SCOUT_PROFILE: str = "precise_v1"
EARLY_TOP_MOVER_SCOUT_TF: tuple[str, ...] = ("15m", "1h")
EARLY_TOP_MOVER_SCOUT_MODE: str = "trend"
EARLY_TOP_MOVER_SCOUT_SLOPE_MIN: float = 0.08
EARLY_TOP_MOVER_SCOUT_ADX_MIN: float = 14.0
EARLY_TOP_MOVER_SCOUT_RSI_MIN: float = 50.0
EARLY_TOP_MOVER_SCOUT_RSI_MAX: float = 75.5
EARLY_TOP_MOVER_SCOUT_VOL_X_MIN: float = 0.55
EARLY_TOP_MOVER_SCOUT_EMA_SEP_MIN_PCT: float = -0.18
EARLY_TOP_MOVER_SCOUT_PRICE_EDGE_MIN_PCT: float = -0.25
EARLY_TOP_MOVER_SCOUT_PRICE_EDGE_MAX_PCT: float = 2.80
EARLY_TOP_MOVER_SCOUT_MACD_MIN_REL: float = -0.00008
EARLY_TOP_MOVER_SCOUT_RECENT_HIGH_LOOKBACK: int = 10
EARLY_TOP_MOVER_SCOUT_RECENT_HIGH_GAP_MAX_PCT: float = 1.10
EARLY_TOP_MOVER_SCOUT_DEDUP_BARS: int = 4
EARLY_TOP_MOVER_SCOUT_SCORE_DEFICIT_MAX: float = 10.0
EARLY_TOP_MOVER_SCOUT_MIN_TODAY_CHANGE_PCT: float = 0.0
EARLY_TOP_MOVER_SCOUT_MIN_FORECAST_RETURN_PCT: float = 0.0
DISCOVERY_SCAN_SEC: int = 300
DISCOVERY_ENTRY_GRACE_BARS: int = 2
DISCOVERY_ENTRY_MAX_SLIPPAGE_PCT: float = 0.45
WAKEUP_SCOUT_ENABLED: bool = True
WAKEUP_SCOUT_PROFILE: str = "wake_up_1m_light15_v1"
WAKEUP_SCOUT_SCAN_SEC: int = 60
WAKEUP_SCOUT_1M_LIMIT: int = 100
WAKEUP_SCOUT_1M_SLOPE_MIN: float = 0.05
WAKEUP_SCOUT_1M_ADX_MIN: float = 14.0
WAKEUP_SCOUT_1M_RSI_MIN: float = 52.0
WAKEUP_SCOUT_1M_RSI_MAX: float = 78.0
WAKEUP_SCOUT_1M_VOL_X_MIN: float = 1.15
WAKEUP_SCOUT_1M_RECENT_HIGH_LOOKBACK: int = 20
WAKEUP_SCOUT_1M_RECENT_HIGH_GAP_MAX_PCT: float = 0.20
WAKEUP_SCOUT_1M_PERSIST_WINDOW: int = 5
WAKEUP_SCOUT_1M_PERSIST_MIN: int = 3
WAKEUP_SCOUT_15M_SLOPE_MIN: float = 0.0
WAKEUP_SCOUT_15M_ADX_MIN: float = 10.0
WAKEUP_SCOUT_15M_RSI_MIN: float = 48.0
WAKEUP_SCOUT_15M_RSI_MAX: float = 80.0
WAKEUP_SCOUT_PRIORITY_BONUS: float = 4.0
WAKEUP_SCOUT_DEDUP_MINUTES: int = 60
AGENT_SOFT_BLOCK_WATCH_ALERTS_ENABLED: bool = True
AGENT_SOFT_BLOCK_RSI_MAX: float = 75.0
AGENT_SOFT_BLOCK_RSI_MIN_VOL_X: float = 3.0
AGENT_SOFT_BLOCK_IMPULSE_MIN_ADX: float = 15.0
AGENT_SOFT_BLOCK_IMPULSE_MAX_RSI: float = 75.0
AGENT_SOFT_BLOCK_IMPULSE_MIN_VOL_X: float = 2.0
AGENT_SOFT_BLOCK_DAILY_RANGE_MAX_PCT: float = 20.0

# RL/report Telegram notifications.
RL_TELEGRAM_REPORTS_ENABLED: bool = True
TOP_GAINER_CRITIC_TELEGRAM_REPORTS_ENABLED: bool = True
TOP_GAINER_CRITIC_TELEGRAM_FINAL_ONLY: bool = True
TOP_GAINER_CRITIC_TOP_N: int = 20
WATCHLIST_TOP_GAINER_GOAL_TELEGRAM_REPORTS_ENABLED: bool = False
RL_TRAIN_TELEGRAM_REPORTS_ENABLED: bool = False

# Post-factum signal quality evaluator. It reads already emitted BUY/SELL
# events and historical candles; it never generates live signals.
SIGNAL_QUALITY_EVALUATOR_ENABLED: bool = True
SIGNAL_QUALITY_EVALUATOR_TELEGRAM_REPORTS_ENABLED: bool = True
SIGNAL_QUALITY_EVALUATOR_TIMEZONE: str = "Europe/Budapest"
SIGNAL_QUALITY_EVALUATOR_RUN_HOUR_LOCAL: int = 0
SIGNAL_QUALITY_EVALUATOR_RUN_MINUTE_LOCAL: int = 15
SIGNAL_QUALITY_EVALUATOR_RUN_WINDOW_MINUTES: int = 45
SIGNAL_QUALITY_EVALUATOR_TIMEFRAMES: tuple[str, ...] = ("15m", "1h")
SIGNAL_QUALITY_EVALUATOR_SOURCE: str = "all"
SIGNAL_QUALITY_EVALUATOR_TOP_MOVERS_N: int = 20
SIGNAL_QUALITY_EVALUATOR_SYMBOLS: tuple[str, ...] = ()
V2_SHADOW_DAILY_SUMMARY_ENABLED: bool = True
V2_SHADOW_DAILY_SUMMARY_TELEGRAM_ENABLED: bool = True
SUSPICIOUS_REENTRY_SCORECARD_ENABLED: bool = True
SUSPICIOUS_REENTRY_SCORECARD_TELEGRAM_ENABLED: bool = True
V2_SHADOW_REALTIME_TELEGRAM_ENABLED: bool = False
V2_SHADOW_SCAN_TIMEOUT_SEC: int = 20
V2_SHADOW_STATUS_EVERY_SCANS: int = 10

# Daily operator learning-progress report. Runs after final previous-day
# top-gainer and signal-quality reports have landed. Reporting only.
LEARNING_PROGRESS_DAILY_REPORT_ENABLED: bool = True
LEARNING_PROGRESS_DAILY_REPORT_TELEGRAM_ENABLED: bool = True
LEARNING_PROGRESS_DAILY_REPORT_TIMEZONE: str = "Europe/Budapest"
LEARNING_PROGRESS_DAILY_REPORT_HOUR_LOCAL: int = 9
LEARNING_PROGRESS_DAILY_REPORT_MINUTE_LOCAL: int = 0
LEARNING_PROGRESS_DAILY_REPORT_WINDOW_MINUTES: int = 60
LEARNING_PROGRESS_FOCUS_SYMBOLS: tuple[str, ...] = ()

# Automatic post-factum feedback from the signal-quality evaluator. The feedback
# layer may only apply narrow, replay-confirmed runtime adjustments. Wider BUY,
# exit, cluster, or replacement rule changes must stay disabled until a replay
# confirms them separately.
SIGNAL_QUALITY_FEEDBACK_ENABLED: bool = True
SIGNAL_QUALITY_FEEDBACK_AUTO_APPLY_COOLDOWN: bool = True
SIGNAL_QUALITY_FEEDBACK_FILE: str = "../.runtime/signal_quality_feedback.json"
SIGNAL_QUALITY_FEEDBACK_MAX_AGE_HOURS: int = 48
SIGNAL_QUALITY_FEEDBACK_MISS_RATE_MIN: float = 0.65
SIGNAL_QUALITY_FEEDBACK_TOP_MOVER_MISSED_MIN: int = 20
SIGNAL_QUALITY_FEEDBACK_FALSE_POSITIVE_MAX: float = 0.22
SIGNAL_QUALITY_FEEDBACK_COOLDOWN_BARS_ON_PRESSURE: int = 2
SIGNAL_QUALITY_FEEDBACK_EXIT_RULES_ENABLED: bool = False
SIGNAL_QUALITY_FEEDBACK_CLUSTER_RULES_ENABLED: bool = False
SIGNAL_QUALITY_FEEDBACK_REPLACEMENT_RULES_ENABLED: bool = False

# Shadow-only lifecycle telemetry for profitable positions. This does not emit
# exits or change trails; it only records potential overextension for later
# evaluator/backtest analysis.
PEAK_RISK_SHADOW_ENABLED: bool = True
PEAK_RISK_SHADOW_THRESHOLD: float = 50.0
PEAK_RISK_RSI_FLOOR: float = 75.0
PEAK_RISK_EDGE_FLOOR_PCT: float = 5.0

# в”Ђв”Ђ Early trend warnings (Stage-1, NO ENTRY) в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
# Stage-1 only sends РїСЂРµРґСѓРїСЂРµР¶РґРµРЅРёСЏ (watch), РЅРµ РѕС‚РєСЂС‹РІР°РµС‚ РїРѕР·РёС†РёСЋ.
EARLY_WARN_COOLDOWN_SEC = 3600   # 1 С‡Р°СЃ: РґРµРґСѓРї РїСЂРµРґСѓРїСЂРµР¶РґРµРЅРёР№ РЅР° РјРѕРЅРµС‚Сѓ/TF
EARLY_WARN_ADX_MIN      = 12.0   # СѓР±СЂР°С‚СЊ СЃРѕРІСЃРµРј РјС‘СЂС‚РІС‹Р№ Р±РѕРєРѕРІРёРє
EARLY_WARN_RSI_MAX      = 82.0   # РЅРµ РґРѕРіРѕРЅСЏС‚СЊ РїРёРєРё
EARLY_WARN_VOLX_MIN     = 1.00   # РјРёРЅРёРјР°Р»СЊРЅР°СЏ Р°РєС‚РёРІРЅРѕСЃС‚СЊ СЂС‹РЅРєР°
EARLY_WARN_VOLJUMP_MIN  = 1.20   # Р°Р»СЊС‚РµСЂРЅР°С‚РёРІРЅС‹Р№ С‚СЂРёРіРіРµСЂ РІРјРµСЃС‚Рѕ volГ—

# в”Ђв”Ђ Exit warnings dedup в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
EXIT_WARN_COOLDOWN_SEC  = 2700   # 45 РјРёРЅСѓС‚: РґРµРґСѓРї exit-warning СЃРѕРѕР±С‰РµРЅРёР№

# Shadow-only suspicious re-entry alerts. These never open positions and never
# bypass production cooldown; they only emit/log "would re-enter" telemetry.
SUSPICIOUS_REENTRY_SHADOW_ENABLED: bool = True
SUSPICIOUS_REENTRY_SHADOW_TELEGRAM_ENABLED: bool = True
SUSPICIOUS_REENTRY_SHADOW_WINDOW_BARS: int = 8
SUSPICIOUS_REENTRY_SHADOW_EXIT_SCORE_MIN: float = 0.68
SUSPICIOUS_REENTRY_SHADOW_MIN_MFE_PCT: float = 1.0
SUSPICIOUS_REENTRY_SHADOW_MIN_CANDIDATE_SCORE: float = 38.0
SUSPICIOUS_REENTRY_SHADOW_MIN_ADX: float = 18.0
SUSPICIOUS_REENTRY_SHADOW_DEDUP_BARS: int = 8
OBSERVABLE_TAIL_SHADOW_ENABLED: bool = True
OBSERVABLE_TAIL_SHADOW_SELECTOR: str = "non_ema_positive_giveback"
OBSERVABLE_TAIL_SHADOW_MIN_PNL_PCT: float = 0.0
OBSERVABLE_TAIL_SHADOW_MIN_GIVEBACK_PCT: float = 0.5
OBSERVABLE_TAIL_SHADOW_SELL_FRACTION: float = 0.5
OBSERVABLE_TAIL_SHADOW_HORIZONS: tuple[int, ...] = (2, 5, 10)
EXIT_LEARNING_EARLY_EXIT_MAX_MINUTES: int = 60
EXIT_LEARNING_POST_EXIT_CONTINUATION_MIN_PCT: float = 1.0


# в”Ђв”Ђ Paper execution (entry quality) в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
PAPER_FEE_BPS = 7.5  # approx taker fee in bps for simulation
PAPER_QUOTE = 50.0  # simulated quote size per entry (USDT)
