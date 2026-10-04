"""Isolated, causal price-forecast research. Never imported by the trading bot."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import time
import urllib.parse
import urllib.request
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ForecastConfig:
    symbols: tuple = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    end_utc: str = "2026-10-04T00:00:00Z"  # exclusive, frozen before evaluation
    history_days: int = 30
    horizon: int = 15
    context: int = 60
    arrival_delay_seconds: int = 2  # assumption, NOT observed historical latency
    fractions: tuple = (0.60, 0.15, 0.10, 0.15)
    eval_stride: int = 60
    calibration_stride: int = 15
    classical_window: int = 720
    models: tuple = ("Persistence", "Ridge", "XGBoost", "ARIMA")
    seed: int = 42
    bootstrap_draws: int = 2000
    min_test_days: int = 10
    complete_test_days_only: bool = True

    def __post_init__(self):
        if (len(self.fractions) != 4 or min(self.fractions) <= 0
                or not np.isclose(sum(self.fractions), 1)):
            raise ValueError("Require four positive split fractions summing to one")
        if min(self.history_days, self.horizon, self.context, self.eval_stride,
               self.calibration_stride, self.classical_window, self.bootstrap_draws) <= 0:
            raise ValueError("Sizes and strides must be positive")
        if min(self.eval_stride, self.calibration_stride) < self.horizon:
            raise ValueError("Use nonoverlapping label paths for evaluation/calibration")
        if self.horizon != 15 or 1440 % self.eval_stride or 1440 % self.calibration_stride:
            raise ValueError("This demo uses h15 and grids dividing a UTC day")
        if self.arrival_delay_seconds < 0:
            raise ValueError("Arrival delay must be nonnegative")
        end = utc(self.end_utc)
        if end != end.floor("min"):
            raise ValueError("Exclusive end must be on the UTC minute grid")
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError("Duplicate symbols")


def utc(value):
    t = pd.Timestamp(value)
    if t.tzinfo is None:
        raise ValueError("Timestamp must have an explicit timezone")
    return t.tz_convert("UTC")


KLINE_COLUMNS = ["open_time", "open", "high", "low", "close", "volume",
                 "close_time", "quote_asset_volume", "number_of_trades",
                 "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"]
OHLCV = ["open", "high", "low", "close", "volume"]
CALENDAR = ["day_sin", "day_cos", "week_sin", "week_cos"]
FEATURES = ([f"ret_{k}" for k in (1, 2, 3, 5, 10, 15, 30, 60)]
            + [f"vol_{k}" for k in (5, 15, 30, 60)]
            + ["range", "body", "volume_log", "volume_change", "volume_z_15",
               "volume_z_60"] + CALENDAR)
STATE_EXOG = ["vol_15", "vol_60", "volume_z_15", "volume_z_60", "ret_15"]


def clean_klines(rows, cfg: ForecastConfig):
    """Fail closed rather than silently altering the forecasting time grid."""
    if not rows or any(not isinstance(r, list) or len(r) != 12 for r in rows):
        raise ValueError("Empty response or changed kline schema")
    d = pd.DataFrame(rows, columns=KLINE_COLUMNS)
    for col in ["open_time", "close_time"]:
        d[col] = pd.to_datetime(pd.to_numeric(d[col], errors="raise"), unit="ms", utc=True)
    numeric = KLINE_COLUMNS[1:6] + KLINE_COLUMNS[7:11]
    d[numeric] = d[numeric].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(d[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite market data")
    duplicates = d[d.duplicated("open_time", keep=False)]
    if not duplicates.empty:
        # Repeating an identical API row is harmless; revised/conflicting rows aren't.
        if (duplicates.groupby("open_time")[d.columns.difference(["open_time"])]
                .nunique(dropna=False).to_numpy() > 1).any():
            raise ValueError("Conflicting duplicate timestamps")
    d = d.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    end = utc(cfg.end_utc)
    start = end - pd.Timedelta(days=cfg.history_days)
    expected = pd.date_range(start, end, freq="min", inclusive="left")
    if len(d) != len(expected) or not pd.DatetimeIndex(d.open_time).equals(expected):
        raise ValueError("Incomplete/extra response or missing one-minute candles")
    if not (d.close_time == d.open_time + pd.Timedelta(minutes=1) - pd.Timedelta(milliseconds=1)).all():
        raise ValueError("Invalid or unclosed candle timestamps")
    if (d.close_time >= end).any():
        raise ValueError("Snapshot contains unclosed candles")
    if (d[["open", "high", "low", "close"]] <= 0).any().any() or (d.volume < 0).any():
        raise ValueError("Invalid prices or volume")
    if ((d.high < d[["open", "close", "low"]].max(axis=1)).any()
            or (d.low > d[["open", "close", "high"]].min(axis=1)).any()):
        raise ValueError("Invalid OHLC relations")
    return d


def fetch_history(symbol, cfg, cache_dir, request=None):
    """Same frozen exclusive end for every symbol; no partial-success fallback."""
    cache = Path(cache_dir) if cache_dir is not None else None
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
    end = utc(cfg.end_utc)
    if end + pd.Timedelta(seconds=cfg.arrival_delay_seconds) > pd.Timestamp.now(tz="UTC"):
        raise ValueError("Snapshot has not become available yet")
    key = end.strftime("%Y%m%dT%H%MZ")
    path = cache / f"{symbol}_1m_{cfg.history_days}d_{key}.json" if cache is not None else None
    if path is not None and path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload["rows"]
        digest = hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()
        if payload["sha256"] != digest or payload["end_utc"] != end.isoformat():
            raise ValueError("Input cache digest/period mismatch")
    else:
        end_ms = end.value // 1_000_000
        cursor = (end - pd.Timedelta(days=cfg.history_days)).value // 1_000_000
        rows = []
        source = "https://data-api.binance.vision/api/v3/klines"
        while cursor < end_ms:
            params = dict(symbol=symbol, interval="1m", startTime=int(cursor),
                          endTime=int(end_ms - 1), limit=1000)
            url = source + "?" + urllib.parse.urlencode(params)
            batch = None
            for attempt in range(3):
                try:
                    if request is None:
                        with urllib.request.urlopen(url, timeout=30) as response:
                            batch = json.load(response)
                    else:
                        batch = request(url)
                    if not isinstance(batch, list) or not batch:
                        raise ValueError("Missing/invalid API page")
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(0.5 * (attempt + 1))
            rows.extend(batch)
            next_cursor = int(batch[-1][0]) + 60_000
            if next_cursor <= cursor:
                raise ValueError("Pagination did not advance")
            cursor = next_cursor
        clean_klines(rows, cfg)  # only persist a complete valid snapshot
        digest = hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()
        payload = dict(symbol=symbol, end_utc=end.isoformat(), sha256=digest, rows=rows,
                       fetched_utc=pd.Timestamp.now(tz="UTC").isoformat(), source=source)
        if path is not None:
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
            tmp.replace(path)
    return clean_klines(rows, cfg), {k: v for k, v in payload.items() if k != "rows"}


def calendar_features(times):
    t = pd.DatetimeIndex(times)
    minute = t.hour * 60 + t.minute
    week = t.dayofweek * 1440 + minute
    return pd.DataFrame(dict(day_sin=np.sin(2*np.pi*minute/1440),
                             day_cos=np.cos(2*np.pi*minute/1440),
                             week_sin=np.sin(2*np.pi*week/10080),
                             week_cos=np.cos(2*np.pi*week/10080)))


def prepare(d, cfg):
    if not (d.open_time.diff().dropna() == pd.Timedelta(minutes=1)).all():
        raise ValueError("Never compress missing minutes")
    x = d.reset_index(drop=True).copy()
    x["time_idx"] = np.arange(len(x))
    x["available_at"] = x.open_time + pd.Timedelta(minutes=1, seconds=cfg.arrival_delay_seconds)
    x["log_close"] = np.log(x.close)
    for k in (1, 2, 3, 5, 10, 15, 30, 60):
        x[f"ret_{k}"] = x.log_close.diff(k)
    for k in (5, 15, 30, 60):
        x[f"vol_{k}"] = x.ret_1.rolling(k).std()
    x["range"] = (x.high - x.low) / x.close
    x["body"] = (x.close - x.open) / x.open
    x["volume_log"] = np.log1p(x.volume)
    x["volume_change"] = x.volume_log.diff()
    for k in (15, 60):
        # A constant volume window is a neutral z-score, not a row deletion.
        mean = x.volume_log.rolling(k).mean()
        std = x.volume_log.rolling(k).std()
        x[f"volume_z_{k}"] = ((x.volume_log - mean) / std.mask(std == 0, 1))
    x[CALENDAR] = calendar_features(x.open_time).to_numpy()
    for h in range(1, cfg.horizon + 1):
        x[f"target_h{h}"] = x.log_close.shift(-h) - x.log_close
    x["label_available_at"] = x.available_at.shift(-cfg.horizon)
    return x


def target_columns(cfg):
    return [f"target_h{h}" for h in range(1, cfg.horizon + 1)]


def split_frames(prepared, cfg):
    start = utc(cfg.end_utc) - pd.Timedelta(days=cfg.history_days)
    minutes = cfg.history_days * 1440
    offsets = [0] + [int(minutes*f) for f in np.cumsum(cfg.fractions)]
    bounds = [start + pd.Timedelta(minutes=k, seconds=cfg.arrival_delay_seconds) for k in offsets]
    valid = np.isfinite(prepared[FEATURES + target_columns(cfg)].to_numpy()).all(axis=1)
    splits = {}
    for i, name in enumerate(("train", "tune", "calibration", "test")):
        mask = (valid & (prepared.available_at >= bounds[i])
                & (prepared.available_at < bounds[i+1])
                & (prepared.label_available_at < bounds[i+1]))
        splits[name] = prepared.loc[mask].copy()
        if len(splits[name]) < cfg.context + cfg.horizon:
            raise ValueError(f"Insufficient {name} support")
    return splits, bounds


def evaluation_grid(frame, stride):
    # Absolute UTC grid, shared across assets; never linspace after dropna.
    minutes = pd.DatetimeIndex(frame.open_time).as_unit("ns").asi8 // 60_000_000_000
    return frame.loc[minutes % stride == 0].copy()


def sequence_inputs(full, rows, cfg):
    a = full[FEATURES].to_numpy(dtype=np.float32)
    out = []
    for idx in rows.time_idx.to_numpy(dtype=int):
        window = a[idx-cfg.context+1:idx+1]
        if idx < cfg.context-1 or len(window) != cfg.context or not np.isfinite(window).all():
            raise ValueError("Invalid sequence context")
        out.append(window)
    return np.stack(out)


class FixedRegressor:
    """All transforms fit on train; validation is exclusively model/epoch selection."""
    def __init__(self, name, train, tune, full, cfg):
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge
        self.cfg, self.name = cfg, name
        if name == "Ridge":
            self.model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        elif name == "XGBoost":
            from xgboost import XGBRegressor
            from sklearn.multioutput import MultiOutputRegressor
            self.model = MultiOutputRegressor(XGBRegressor(
                n_estimators=250, max_depth=5, learning_rate=0.04, subsample=0.85,
                colsample_bytree=0.85, reg_lambda=2, objective="reg:squarederror",
                n_jobs=2, random_state=cfg.seed), n_jobs=1)
        else:
            raise ValueError(name)
        self.model.fit(train[FEATURES], train[target_columns(cfg)])

    def predict(self, full, rows):
        return self.model.predict(rows[FEATURES])


class LSTMPolicy:
    def __init__(self, name, train, tune, full, cfg):
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.preprocessing import StandardScaler
        self.cfg, self.torch = cfg, torch
        torch.manual_seed(cfg.seed)
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        self.xscale = StandardScaler().fit(train[FEATURES])
        self.yscale = StandardScaler().fit(train[target_columns(cfg)])
        def data(rows):
            x = sequence_inputs(full, rows, cfg)
            x = self.xscale.transform(x.reshape(-1, len(FEATURES))).reshape(x.shape)
            y = self.yscale.transform(rows[target_columns(cfg)])
            return TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
        train = train.loc[train.time_idx >= 60 + cfg.context - 1]
        training = DataLoader(data(train), batch_size=256, shuffle=False)
        validation = DataLoader(data(tune), batch_size=256, shuffle=False)
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(len(FEATURES), 32, batch_first=True)
                self.head = nn.Linear(32, cfg.horizon)
            def forward(self, x):
                return self.head(self.lstm(x)[0][:, -1])
        self.model = Net()
        opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        best, state = math.inf, None
        self.epoch_losses = []
        for epoch in range(6):
            self.model.train()
            for xb, yb in training:
                opt.zero_grad()
                loss = nn.functional.mse_loss(self.model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                opt.step()
            self.model.eval()
            total, count = 0.0, 0
            with torch.no_grad():
                for xb, yb in validation:
                    total += nn.functional.mse_loss(self.model(xb), yb, reduction="sum").item()
                    count += yb.numel()
            val = total/count
            self.epoch_losses.append(val)
            if val < best:
                best = val
                state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict(state)
        self.model.eval()

    def predict(self, full, rows):
        x = sequence_inputs(full, rows, self.cfg)
        x = self.xscale.transform(x.reshape(-1, len(FEATURES))).reshape(x.shape)
        with self.torch.no_grad():
            pred = self.model(self.torch.tensor(x, dtype=self.torch.float32)).numpy()
        return self.yscale.inverse_transform(pred)


class ClassicalPolicy:
    def __init__(self, name, train, tune, full, cfg):
        if name == "Prophet":
            import prophet  # fail before scoring if unavailable
        else:
            import statsmodels
        self.name, self.cfg = name, cfg
        self.warnings = []
        self.warning_counts = {}

    def predict(self, full, rows):
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        cfg = self.cfg
        preds = []
        for _, row in rows.iterrows():
            idx = int(row.time_idx)
            hist = full.iloc[max(60, idx-cfg.classical_window+1):idx+1].copy()
            if len(hist) < 120 or hist.available_at.max() > row.available_at:
                raise ValueError("Insufficient/noncausal classical history")
            # Tiny minute-return variance makes raw-log MLE poorly conditioned.
            # Center/scale only from the causal history of THIS origin.
            scale = max(float(hist.log_close.std()), 1e-6)
            y = (hist.log_close.to_numpy()-float(row.log_close))/scale
            with warnings.catch_warnings(record=True) as observed:
                warnings.simplefilter("always")
                if self.name == "ARIMA":
                    fitted = ARIMA(y, order=(1, 1, 0), trend="n").fit(method="yule_walker")
                    forecast = np.asarray(fitted.get_forecast(cfg.horizon).predicted_mean)
                elif self.name in ("SARIMA", "SARIMAX"):
                    exog, future_exog = None, None
                    if self.name == "SARIMAX":
                        exog = hist[CALENDAR + STATE_EXOG]
                        future_exog = calendar_features(pd.date_range(
                            row.open_time + pd.Timedelta(minutes=1), periods=cfg.horizon, freq="min"))
                        for col in STATE_EXOG:
                            future_exog[col] = float(row[col])
                    fitted = SARIMAX(y, exog=exog, order=(1, 1, 0),
                        seasonal_order=(1, 0, 0, 60), trend="n").fit(disp=False, maxiter=100)
                    forecast = np.asarray(fitted.get_forecast(cfg.horizon, exog=future_exog).predicted_mean)
                elif self.name == "Prophet":
                    from prophet import Prophet
                    # Two complete daily periods minimum; 12h cannot identify daily seasonality.
                    if cfg.classical_window < 2880:
                        raise ValueError("Prophet daily seasonality requires window >= 2880 minutes")
                    model = Prophet(daily_seasonality=True, weekly_seasonality=False,
                                    yearly_seasonality=False, uncertainty_samples=0)
                    model.fit(pd.DataFrame(dict(ds=hist.open_time.dt.tz_localize(None), y=y)))
                    forecast = model.predict(pd.DataFrame(dict(ds=pd.date_range(
                        row.open_time.tz_localize(None) + pd.Timedelta(minutes=1),
                        periods=cfg.horizon, freq="min")))).yhat.to_numpy()
                    fitted = None
                else:
                    raise ValueError(self.name)
            for warning in observed:
                message = str(warning.message)
                self.warning_counts[message] = self.warning_counts.get(message,0)+1
                if message not in self.warnings:
                    self.warnings.append(message)
            # Yule-Walker estimates AR(1) directly, without iterative MLE.
            if fitted is not None and self.name != "ARIMA" and not fitted.mle_retvals.get("converged", False):
                raise ValueError(f"{self.name} did not converge at {row.open_time}")
            if fitted is not None and not np.isfinite(fitted.params).all():
                raise ValueError(f"{self.name} returned nonfinite parameters")
            preds.append(forecast*scale)
        return np.asarray(preds)


def finite_sample_widths(actual, predictions, nominal=0.90):
    errors = np.abs(np.asarray(actual) - np.asarray(predictions))
    if errors.ndim != 2 or errors.shape != np.asarray(predictions).shape or not np.isfinite(errors).all():
        raise ValueError("Invalid calibration residuals")
    n = len(errors)
    rank = math.ceil((n+1)*nominal)
    if not 0 < nominal < 1 or rank > n:
        raise ValueError("Insufficient calibration support for finite-sample quantile")
    return np.sort(errors, axis=0)[rank-1]


def choose_model(tuning_predictions, actual):
    """Receives no test/calibration labels; persistence wins exact ties."""
    ordered = sorted(tuning_predictions, key=lambda name: (name != "Persistence", name))
    losses = {name: float(np.mean(np.abs(actual[:, -1] - tuning_predictions[name][:, -1])))
              for name in ordered}
    if not losses or not all(np.isfinite(list(losses.values()))):
        raise ValueError("Invalid tuning losses")
    return min(ordered, key=lambda name: losses[name]), losses


def paired_day_bootstrap(rows, actual, predictions, cfg):
    baseline_error = np.abs(actual[:, -1])
    model_error = np.abs(actual[:, -1] - predictions[:, -1])
    groups = pd.DataFrame(dict(day=rows.open_time.dt.floor("D").to_numpy(),
                               delta=baseline_error-model_error, base=baseline_error))
    days = groups.groupby("day", sort=True).agg(delta=("delta", "sum"),
                                                base=("base", "sum"), n=("delta", "size"))
    out = dict(n_time_blocks=len(days), block="UTC day", bootstrap_draws=cfg.bootstrap_draws,
               ci95_improvement_pct=None, verdict="UNKNOWN")
    if len(days) < cfg.min_test_days or days.base.sum() <= 0:
        return out
    rng = np.random.default_rng(cfg.seed)
    samples = rng.integers(0, len(days), size=(cfg.bootstrap_draws, len(days)))
    delta = days.delta.to_numpy()[samples].sum(axis=1)
    denom = days.base.to_numpy()[samples].sum(axis=1)
    if (denom <= 0).any():
        return out
    lo, hi = np.quantile(100*delta/denom, [0.025, 0.975])
    out.update(ci95_improvement_pct=[float(lo), float(hi)],
               verdict="SUPPORTED_DIAGNOSTIC" if lo > 0 else "NOT_PROVEN")
    return out


def score(symbol, name, rows, pred, widths, cfg):
    actual = rows[target_columns(cfg)].to_numpy()
    pred = np.asarray(pred)
    if pred.shape != actual.shape or not len(rows) or not np.isfinite(pred).all():
        raise ValueError("Partial/nonfinite/shape-mismatched predictions are not scored")
    error, base = np.abs(actual-pred), np.abs(actual)
    close = rows.close.to_numpy()[:, None]
    price_actual, price_pred = close*np.exp(actual), close*np.exp(pred)
    baseline_sum, model_sum = float(base[:, -1].sum()), float(error[:, -1].sum())
    result = dict(symbol=symbol, model=name, n_origins=len(rows), n_path_points=actual.size,
                  test_start=rows.open_time.iloc[0].isoformat(), test_end=rows.open_time.iloc[-1].isoformat(),
                  MAE_h15_return=float(error[:, -1].mean()), baseline_MAE_h15_return=float(base[:, -1].mean()),
                  improvement_numerator=baseline_sum-model_sum, improvement_denominator=baseline_sum,
                  improvement_pct=100*(baseline_sum-model_sum)/baseline_sum if baseline_sum > 0 else None,
                  MAE_h15_USDT=float(np.abs(price_actual[:, -1]-price_pred[:, -1]).mean()),
                  MAE_by_horizon=error.mean(axis=0).tolist(),
                  RMSE_by_horizon=np.sqrt(((actual-pred)**2).mean(axis=0)).tolist())
    # No-change forecasts are abstentions; they do not receive a misleading sign-accuracy score.
    directional = pred[:, -1] != 0
    correct = int(((np.sign(pred[:, -1]) == np.sign(actual[:, -1])) & directional).sum())
    result.update(direction_correct=correct, direction_n=int(directional.sum()),
                  direction_accuracy=correct/int(directional.sum()) if directional.any() else None,
                  up_base_count=int((actual[:, -1] > 0).sum()), direction_population_n=len(rows))
    if widths is not None:
        q = np.asarray(widths)
        if q.shape != (cfg.horizon,) or not np.isfinite(q).all() or (q < 0).any():
            raise ValueError("Invalid per-horizon interval widths")
        lo, hi = pred-q, pred+q
        covered = ((actual >= lo) & (actual <= hi)).sum(axis=0)
        interval_score = (hi-lo) + 20*np.maximum(lo-actual, 0) + 20*np.maximum(actual-hi, 0)
        result.update(interval_kind="empirical_log_return_residual_90",
                      PI90_covered_by_horizon=covered.tolist(), PI90_denominator=len(rows),
                      PI90_coverage_by_horizon=(covered/len(rows)).tolist(),
                      PI90_interval_score_by_horizon=interval_score.mean(axis=0).tolist(),
                      PI90_mean_width_USDT_by_horizon=(close*(np.exp(hi)-np.exp(lo))).mean(axis=0).tolist())
    result.update(paired_day_bootstrap(rows, actual, pred, cfg))
    return result


def build_policy(name, train, tune, full, cfg):
    if name in ("Ridge", "XGBoost"):
        return FixedRegressor(name, train, tune, full, cfg)
    if name == "LSTM":
        return LSTMPolicy(name, train, tune, full, cfg)
    if name in ("ARIMA", "SARIMA", "SARIMAX", "Prophet"):
        return ClassicalPolicy(name, train, tune, full, cfg)
    if name == "TFT":
        return TFTPolicy(name, train, tune, full, cfg)
    raise ValueError(name)


def run_experiment(market, cfg):
    """Freeze selection on tuning before ever predicting/scoring final test."""
    prepared = {s: prepare(market[s], cfg) for s in cfg.symbols}
    split_map = {s: split_frames(prepared[s], cfg)[0] for s in cfg.symbols}
    grids = {s: {stage: evaluation_grid(sp[stage], cfg.calibration_stride if stage == "calibration" else cfg.eval_stride)
                 for stage in ("tune", "calibration", "test")} for s, sp in split_map.items()}
    if cfg.complete_test_days_only:
        # Drop boundary days before fitting, without examining their values.
        for s in cfg.symbols:
            rows = grids[s]["test"]
            counts = rows.groupby(rows.open_time.dt.floor("D")).size()
            eligible = counts[counts == 1440 // cfg.eval_stride].index
            grids[s]["test"] = rows.loc[rows.open_time.dt.floor("D").isin(eligible)].copy()
            if grids[s]["test"].empty:
                raise ValueError("No complete test days")
    for stage in ("tune", "calibration", "test"):
        reference = grids[cfg.symbols[0]][stage].open_time.reset_index(drop=True)
        for s in cfg.symbols:
            if not grids[s][stage].open_time.reset_index(drop=True).equals(reference):
                raise ValueError("Asset grids differ; a comparable benchmark cannot be built")
    policies, tuning, widths, status, choices, results, predictions = {}, {}, {}, [], {}, [], {}
    for s in cfg.symbols:
        sp, full, grid = split_map[s], prepared[s], grids[s]
        policies[s], tuning[s], widths[s], predictions[s] = {}, {}, {}, {}
        for name in dict.fromkeys(("Persistence",)+cfg.models):
            policy = None
            try:
                if name == "Persistence":
                    tune_pred = np.zeros((len(grid["tune"]), cfg.horizon))
                else:
                    policy = build_policy(name, sp["train"], sp["tune"], full, cfg)
                    tune_pred = policy.predict(full, grid["tune"])
                if tune_pred.shape != (len(grid["tune"]), cfg.horizon) or not np.isfinite(tune_pred).all():
                    raise ValueError("Invalid/partial tuning predictions")
                policies[s][name], tuning[s][name] = policy, tune_pred
                status.append(dict(symbol=s, model=name, stage="fit_tune", status="COMPLETE",
                                   n_train=len(sp["train"]), n_tune=len(grid["tune"]),
                                   warnings=getattr(policy, "warnings", [])))
            except Exception as exc:
                status.append(dict(symbol=s, model=name, stage="fit_tune", status="UNAVAILABLE" if
                    isinstance(exc, ImportError) else "FAIL", reason=f"{type(exc).__name__}: {exc}",
                    warnings=getattr(policy, "warnings", [])))
        choices[s], tuning_losses = choose_model(tuning[s], grid["tune"][target_columns(cfg)].to_numpy())
        status.append(dict(symbol=s, stage="selection_locked_before_test", selected=choices[s],
                           tuning_MAE_h15_return=tuning_losses))
    # Selection is now frozen for every asset. Test failures never trigger test-driven reselection.
    for s in cfg.symbols:
        grid, full = grids[s], prepared[s]
        for name, policy in policies[s].items():
            try:
                cal_pred = np.zeros((len(grid["calibration"]), cfg.horizon)) if policy is None else policy.predict(full, grid["calibration"])
                widths[s][name] = finite_sample_widths(grid["calibration"][target_columns(cfg)].to_numpy(), cal_pred)
                pred = np.zeros((len(grid["test"]), cfg.horizon)) if policy is None else policy.predict(full, grid["test"])
                row = score(s, name, grid["test"], pred, widths[s][name], cfg)
                row["selected_before_test"] = choices[s] == name
                row["n_calibration"] = len(grid["calibration"])
                results.append(row)
                predictions[s][name] = pred
                status.append(dict(symbol=s, model=name, stage="test", status="COMPLETE", n_origins=len(pred),
                                   warnings=getattr(policy, "warnings", [])))
            except Exception as exc:
                status.append(dict(symbol=s, model=name, stage="test", status="FAIL", reason=f"{type(exc).__name__}: {exc}",
                                   warnings=getattr(policy, "warnings", [])))
        print(s, "selected:", choices[s], "test origins:", len(grid["test"]), flush=True)
    versions = {}
    for pkg in ("numpy", "pandas", "scipy", "scikit-learn", "xgboost", "statsmodels", "torch", "prophet", "pytorch-forecasting"):
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "unavailable"
    metadata = dict(config=asdict(cfg), versions=versions, python=platform.python_version(),
                    source_sha256=globals().get("SOURCE_SHA256") or (hashlib.sha256(Path(__file__).read_bytes()).hexdigest() if "__file__" in globals() else None),
                    status="research_only", actual_arrival_times="UNKNOWN: assumed close+delay",
                    policy_comparison="fixed supervised fit vs causal rolling classical refit",
                    ARIMA_estimator="ARIMA(1,1,0), train-origin scaling, Yule-Walker on differences",
                    split_counts={s: {k:len(v) for k,v in sp.items()} for s,sp in split_map.items()},
                    optional_models_not_requested=[m for m in ("SARIMA", "SARIMAX", "Prophet", "LSTM", "TFT") if m not in cfg.models])
    return dict(results=results, status=status, metadata=metadata, choices=choices,
                prepared=prepared, grids=grids, policies=policies, predictions=predictions, widths=widths)


def comparable_ranking(results, cfg):
    table = pd.DataFrame(results)
    rows = []
    for name, group in table.groupby("model"):
        if set(group.symbol) != set(cfg.symbols):
            continue  # do not rank a model on a conveniently successful subset
        values = group.improvement_pct.to_numpy(dtype=float)
        rows.append(dict(model=name, n_assets=len(group),
                         mean_improvement_pct=float(values.mean()) if np.isfinite(values).all() else None,
                         mean_MAE_h15_return=float(group.MAE_h15_return.mean())))
    return pd.DataFrame(rows).sort_values("mean_MAE_h15_return")


def expanding_window_check(experiment, cfg, model_names=("Ridge",)):
    """Three expanding folds inside TRAIN only; final test stays sealed.

    This diagnostic is preregistered, not used to tune the final test results.
    Add XGBoost explicitly before a fresh run if a boosting stability claim is needed.
    """
    output = []
    for symbol in cfg.symbols:
        full = experiment["prepared"][symbol]
        train = split_frames(full, cfg)[0]["train"]
        lo, hi = train.available_at.min(), train.available_at.max()
        edges = [lo+(hi-lo)*f for f in (0.4, 0.6, 0.8, 1.0)]
        for fold in range(3):
            start, end = edges[fold], edges[fold+1]
            past = train.loc[train.label_available_at < start]
            valid = train.loc[(train.available_at >= start) & (train.label_available_at < end)]
            rows = evaluation_grid(valid, cfg.eval_stride)
            for name in ("Persistence",)+tuple(model_names):
                try:
                    model = None if name == "Persistence" else build_policy(name,past,valid,full,cfg)
                    pred = np.zeros((len(rows),cfg.horizon)) if model is None else model.predict(full,rows)
                    result = score(symbol,name,rows,pred,None,cfg)
                    result.update(fold=fold+1, n_train=len(past), stage="train_only_expanding_cv")
                    output.append(result)
                except Exception as exc:
                    output.append(dict(symbol=symbol,model=name,fold=fold+1,status="FAIL",reason=str(exc)))
    return output


def illustrative_forecast(experiment, symbol, cfg):
    """Latest CLOSED snapshot, frozen tuning choice, no test-driven refitting."""
    full = experiment["prepared"][symbol]
    latest = full.iloc[[-1]]
    if not np.isfinite(latest[FEATURES].to_numpy()).all():
        raise ValueError("Latest snapshot lacks finite causal features")
    name = experiment["choices"][symbol]
    failed = any(r.get("symbol") == symbol and r.get("model") == name
                 and r.get("stage") == "test" and r.get("status") == "FAIL" for r in experiment["status"])
    if failed:
        raise ValueError("Selected policy failed test; investigate instead of silently reselecting")
    model = experiment["policies"][symbol][name]
    pred = np.zeros((1, cfg.horizon)) if model is None else model.predict(full, latest)
    q = experiment["widths"][symbol][name]
    close = float(latest.close.iloc[0])
    return pd.DataFrame(dict(target_close_at=pd.date_range(latest.open_time.iloc[0] + pd.Timedelta(minutes=2),
                              periods=cfg.horizon, freq="min"),
                             predicted_close=close*np.exp(pred[0]), PI90_lower=close*np.exp(pred[0]-q),
                             PI90_upper=close*np.exp(pred[0]+q), model=name,
                             issued_at=latest.available_at.iloc[0], status="illustrative_frozen_snapshot"))


class TFTPolicy:
    """Optional single-asset TFT; decoder contains calendar + origin-frozen state only.

    Train/tune decoder labels obey the same UTC cutoffs as the other models.
    Inference supplies no realized future unknown variables, even as dummy input.
    Its interval is calibrated on the separate calibration split like all models.
    """
    def __init__(self, name, train, tune, full, cfg):
        import torch
        import lightning.pytorch as pl
        from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
        from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
        from pytorch_forecasting.data import GroupNormalizer
        from pytorch_forecasting.metrics import QuantileLoss
        import tempfile
        self.cfg, self.torch, self.dataset_type = cfg, torch, TimeSeriesDataSet
        torch.set_num_threads(2)
        pl.seed_everything(cfg.seed, workers=True)
        d = full.loc[np.isfinite(full[FEATURES].to_numpy()).all(axis=1)].copy()
        d["series"] = "asset"
        d["step"] = d.time_idx.astype(int)
        # Last close used as a decoder target is before next split's first origin.
        train_end = int(train.time_idx.max()) + cfg.horizon
        tune_start, tune_end = int(tune.time_idx.min())+1, int(tune.time_idx.max())+cfg.horizon
        training_frame = d.loc[d.step <= train_end]
        training = TimeSeriesDataSet(training_frame, time_idx="step", target="close", group_ids=["series"],
            max_encoder_length=cfg.context, min_encoder_length=cfg.context,
            max_prediction_length=cfg.horizon, min_prediction_length=cfg.horizon,
            time_varying_known_reals=CALENDAR,
            time_varying_unknown_reals=["close"] + [f for f in FEATURES if f not in CALENDAR],
            target_normalizer=GroupNormalizer(groups=["series"]), allow_missing_timesteps=False)
        tuning_frame = d.loc[(d.step >= tune_start-cfg.context) & (d.step <= tune_end)]
        validation = TimeSeriesDataSet.from_dataset(training, tuning_frame,
            min_prediction_idx=tune_start, stop_randomization=True, predict=False)
        model = TemporalFusionTransformer.from_dataset(training, learning_rate=1e-3, hidden_size=16,
            attention_head_size=2, dropout=0.15, hidden_continuous_size=8, loss=QuantileLoss(),
            reduce_on_plateau_patience=2)
        with tempfile.TemporaryDirectory(prefix="forecast-tft-") as root:
            checkpoint = ModelCheckpoint(dirpath=root, monitor="val_loss", mode="min", save_top_k=1)
            trainer = pl.Trainer(max_epochs=5, accelerator="cpu", devices=1, deterministic=True,
                gradient_clip_val=0.1, logger=False, enable_progress_bar=False,
                callbacks=[checkpoint, EarlyStopping(monitor="val_loss", patience=2, mode="min")])
            trainer.fit(model, training.to_dataloader(train=True, batch_size=128, num_workers=0),
                         validation.to_dataloader(train=False, batch_size=128, num_workers=0))
            if not checkpoint.best_model_path:
                raise ValueError("No validated TFT checkpoint")
            self.model = TemporalFusionTransformer.load_from_checkpoint(checkpoint.best_model_path)
        self.training = training

    def predict(self, full, rows):
        out = []
        for _, row in rows.iterrows():
            idx = int(row.time_idx)
            hist = full.iloc[idx-self.cfg.context+1:idx+1].copy()
            if len(hist) != self.cfg.context or not np.isfinite(hist[FEATURES].to_numpy()).all():
                raise ValueError("TFT missing contiguous encoder context")
            # Create future rows from the origin, NEVER slice actual future market rows.
            future = pd.DataFrame([row.to_dict() for _ in range(self.cfg.horizon)])
            future["open_time"] = pd.date_range(row.open_time+pd.Timedelta(minutes=1), periods=self.cfg.horizon, freq="min")
            future["time_idx"] = np.arange(idx+1, idx+self.cfg.horizon+1)
            future[CALENDAR] = calendar_features(future.open_time).to_numpy()
            frame = pd.concat([hist, future], ignore_index=True)
            frame = frame[["open_time", "close", "time_idx"]+FEATURES].copy()
            frame["series"], frame["step"] = "asset", frame.time_idx.astype(int)
            dataset = self.dataset_type.from_dataset(self.training, frame, predict=True, stop_randomization=True)
            result = self.model.predict(dataset.to_dataloader(train=False, batch_size=1, num_workers=0),
                                        mode="prediction", return_index=True)
            if int(result.index.step.iloc[0]) != idx+1:
                raise ValueError("TFT decoder timestamp mismatch")
            price = result.output.detach().cpu().numpy().reshape(-1)
            if price.shape != (self.cfg.horizon,) or (price <= 0).any():
                raise ValueError("Invalid TFT prediction path")
            out.append(np.log(price/row.close))
        return np.stack(out)


def save_evidence(experiment, manifest, path):
    payload = {key: experiment[key] for key in ("results", "status", "metadata", "choices")}
    payload["inputs"] = manifest
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


class ForecastService:
    """Frozen models, fresh closed candles, serialized refresh, explicit expiry.

    Unproven selected models remain research-only; the serving policy uses the
    predeclared persistence fallback. A new benchmark/release is a separate run.
    """
    def __init__(self, experiment, cfg, token, fetcher=fetch_history, clock=None):
        import threading
        if not token or len(token) < 24:
            raise ValueError("Set a separate FORECAST_API_TOKEN (at least 24 characters)")
        self.experiment, self.cfg, self.token, self.fetcher = experiment, cfg, token, fetcher
        self.clock = clock or (lambda: pd.Timestamp.now(tz="UTC"))
        self.lock = threading.Lock()
        self.snapshots, self.last_error = {}, None
        self.pending_labels, self.quality, self.drift = {}, [], {}
        self.started = self.clock()
        self.max_age_seconds, self.release_max_age_days = 90, 7
        self.release_time = utc(cfg.end_utc)
        self.active = {}
        for s in cfg.symbols:
            selected = experiment["choices"][s]
            row = next((r for r in experiment["results"] if r["symbol"] == s and r["model"] == selected), None)
            self.active[s] = selected if row and row["verdict"] == "SUPPORTED_DIAGNOSTIC" else "Persistence"
        # This statistical gate only serves forecasts. It is never trading approval.

    def refresh(self, cache_dir):
        import logging
        from dataclasses import replace
        now = utc(self.clock())
        if now < self.release_time or now-self.release_time > pd.Timedelta(days=self.release_max_age_days):
            self.last_error = "release_expired_or_clock_invalid"
            raise ValueError(self.last_error)
        end = (now-pd.Timedelta(seconds=self.cfg.arrival_delay_seconds)).floor("min")
        live_cfg = replace(self.cfg, end_utc=end.isoformat(), history_days=1)
        if not self.lock.acquire(blocking=False):
            return False
        try:
            pending = {}
            new_histories = {}
            for s in self.cfg.symbols:
                # Online data stays bounded in memory. Immutable disk cache is for research only.
                market, manifest = self.fetcher(s, live_cfg, None)
                full = prepare(market, live_cfg)
                latest = full.iloc[[-1]]
                new_histories[s] = market
                name = self.active[s]
                model = self.experiment["policies"][s][name]
                pred = np.zeros((1, self.cfg.horizon)) if model is None else model.predict(full, latest)
                if pred.shape != (1, self.cfg.horizon) or not np.isfinite(pred).all():
                    raise ValueError("Nonfinite live prediction")
                q = self.experiment["widths"][s][name]
                issued = latest.available_at.iloc[0]
                if issued > now or (now-issued).total_seconds() > self.max_age_seconds:
                    raise ValueError("Stale/future input snapshot")
                prices = float(latest.close.iloc[0])*np.exp(pred[0])
                pending[s] = dict(symbol=s, model=name, research_selected=self.experiment["choices"][s],
                    fallback=name != self.experiment["choices"][s], input_available_at=issued.isoformat(),
                    input_end_utc=end.isoformat(), input_sha256=manifest["sha256"],
                    data_received_at=manifest.get("fetched_utc",utc(self.clock()).isoformat()),
                    target_close_at=[(end+pd.Timedelta(minutes=h)).isoformat() for h in range(1,self.cfg.horizon+1)],
                    predicted_close=prices.tolist(),
                    PI90_lower=(prices*np.exp(-q)).tolist(), PI90_upper=(prices*np.exp(q)).tolist(),
                    interval_kind="historical_empirical_90_not_guaranteed_under_drift",
                    release_end_utc=self.release_time.isoformat(), actual_arrival_times="not_measured")
                pending[s]["origin_close"] = float(latest.close.iloc[0])
            # Atomic replacement: no mixed-time snapshots after a partial network failure.
            published = utc(self.clock())
            if any(utc(row["target_close_at"][0]) <= published for row in pending.values()):
                raise ValueError("Inference missed first-horizon deadline")
            for row in pending.values():
                row["issued_at"] = published.isoformat()
            self.snapshots, self.last_error = pending, None
            self.observe_matured(new_histories, published)
            for s,row in pending.items():
                self.pending_labels.setdefault((s,row["input_end_utc"]),row)
            self.quality = self.quality[-1000:]
            logging.getLogger("forecast_service").info(json.dumps(dict(event="refresh_complete", end=end.isoformat(),
                n_symbols=len(pending), elapsed_seconds=(utc(self.clock())-now).total_seconds())))
            return True
        except Exception as exc:
            self.last_error = type(exc).__name__
            logging.getLogger("forecast_service").error(json.dumps(dict(event="refresh_failed", error=self.last_error)))
            raise
        finally:
            self.lock.release()

    def observe_matured(self, histories, now):
        for symbol,market in histories.items():
            if "prepared" in self.experiment:
                train=split_frames(self.experiment["prepared"][symbol],self.cfg)[0]["train"]
                recent=prepare(market,self.cfg).iloc[-1]
                std=train[FEATURES].std().replace(0,1)
                z=np.abs((recent[FEATURES]-train[FEATURES].mean())/std)
                self.drift[symbol]=dict(n_features_over_5_train_std=int((z>5).sum()),n_features=len(FEATURES))
            prices=market.set_index("open_time").close
            for key,prior in list(self.pending_labels.items()):
                if prior["symbol"] != symbol:
                    continue
                target=utc(prior["target_close_at"][-1])-pd.Timedelta(minutes=1)
                if target+pd.Timedelta(minutes=1,seconds=self.cfg.arrival_delay_seconds) > now:
                    continue
                if target not in prices.index:
                    receipt=dict(symbol=symbol,status="UNKNOWN_missing_label",issued_at=prior["issued_at"])
                else:
                    actual=float(prices.loc[target])
                    receipt=dict(symbol=symbol,status="MATURED",issued_at=prior["issued_at"],model=prior["model"],
                        error_h15_return=abs(math.log(actual/prior["predicted_close"][-1])),
                        baseline_error_h15_return=abs(math.log(actual/prior["origin_close"])),
                        covered_h15=prior["PI90_lower"][-1]<=actual<=prior["PI90_upper"][-1])
                self.quality.append(receipt)
                del self.pending_labels[key]

    def forecast(self, symbol):
        if symbol not in self.cfg.symbols:
            raise KeyError(symbol)
        row = self.snapshots.get(symbol)
        now = utc(self.clock())
        if (not row or now-self.release_time > pd.Timedelta(days=self.release_max_age_days)
                or utc(row["target_close_at"][0]) <= now
                or not 0 <= (now-utc(row["input_available_at"])).total_seconds() <= self.max_age_seconds):
            raise ValueError("No fresh, unexpired forecast")
        return row


def create_app(experiment, cfg, cache_dir="forecast_online_cache", token=None, fetcher=fetch_history, clock=None):
    """Serve with uvicorn, one worker; TLS/rate limits supplied by deployment."""
    import asyncio
    import hmac
    import os
    from contextlib import asynccontextmanager
    from fastapi import FastAPI, Header, HTTPException
    service = ForecastService(experiment, cfg, token or os.environ.get("FORECAST_API_TOKEN"), fetcher, clock)
    @asynccontextmanager
    async def lifespan(app):
        await asyncio.to_thread(service.refresh, cache_dir)  # no ready state until complete
        stop = asyncio.Event()
        async def updater():
            while not stop.is_set():
                try:
                    await asyncio.wait_for(stop.wait(), timeout=30)
                except asyncio.TimeoutError:
                    try:
                        await asyncio.to_thread(service.refresh, cache_dir)
                    except Exception:
                        pass  # logged; endpoint freshness gate will reject stale values
        task = asyncio.create_task(updater())
        try:
            yield
        finally:
            stop.set()
            await task
    app = FastAPI(title="Causal BTC/ETH/SOL Forecast Demo", lifespan=lifespan)
    app.state.forecast_service = service
    @app.get("/health")
    def health():
        ready = all(s in service.snapshots for s in cfg.symbols)
        if ready:
            try:
                for s in cfg.symbols:
                    service.forecast(s)
            except ValueError:
                ready = False
        if not ready:
            raise HTTPException(status_code=503, detail="Forecast not ready or stale")
        mature=[r for r in service.quality if r["status"] == "MATURED"]
        return dict(status="ready",n_symbols=len(service.snapshots),last_refresh_error=service.last_error,
                    drift=service.drift,matured_forecasts=len(mature),
                    h15_coverage_numerator=sum(r["covered_h15"] for r in mature),h15_coverage_denominator=len(mature),
                    h15_coverage=sum(r["covered_h15"] for r in mature)/len(mature) if mature else None)
    @app.get("/forecast/{symbol}")
    def forecast(symbol: str, x_api_key: str = Header(default="")):
        if not hmac.compare_digest(x_api_key, service.token):
            raise HTTPException(status_code=401, detail="Invalid API key")
        try:
            return service.forecast(symbol)
        except KeyError:
            raise HTTPException(status_code=404, detail="Unsupported symbol")
        except ValueError:
            raise HTTPException(status_code=503, detail="Forecast stale or release expired")
    return app


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end-utc", default=ForecastConfig.end_utc)
    parser.add_argument("--history-days", type=int, default=30)
    parser.add_argument("--cache", default=".runtime/forecast_review/cache")
    parser.add_argument("--output", default=".runtime/forecast_review/benchmark.json")
    parser.add_argument("--models", default="Persistence,Ridge,XGBoost,ARIMA")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    cfg = ForecastConfig(end_utc=args.end_utc, history_days=args.history_days, models=tuple(args.models.split(",")))
    market, manifest = {}, {}
    for symbol in cfg.symbols:
        market[symbol], manifest[symbol] = fetch_history(symbol, cfg, args.cache)
        print(symbol, len(market[symbol]), manifest[symbol]["sha256"], flush=True)
    experiment = run_experiment(market, cfg)
    save_evidence(experiment, manifest, args.output)
    print(comparable_ranking(experiment["results"], cfg).to_string(index=False), flush=True)
    print("Saved:", args.output, flush=True)
    if args.serve:
        import uvicorn
        import logging
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        uvicorn.run(create_app(experiment, cfg, cache_dir=str(Path(args.cache).parent/"online")),
                    host=args.host, port=8000, workers=1)


if __name__ == "__main__":
    main()
