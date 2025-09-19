from __future__ import annotations
import os
from typing import Iterable, Tuple
import numpy as np
import pandas as pd
from datetime import timedelta
try:
    import holidays as _hol
except Exception:
    _hol = None


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def consolidate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nomes das colunas esperadas a partir do dataset consolidado.

    Espera colunas (case-insensitive):
    - internal_store_id -> pdv_id
    - internal_product_id -> produto_id
    - transaction_date -> dt_transacao (datetime)
    - quantity -> quantity (num)

    Mantém demais colunas para possível enriquecimento.
    """
    df = standardize_columns(df)
    rename_map = {
        "internal_store_id": "pdv_id",
        "internal_product_id": "produto_id",
        "transaction_date": "dt_transacao",
        "reference_date": "dt_referencia",
    }
    df = df.rename(columns=rename_map)
    # Tipagem
    if "pdv_id" in df.columns:
        df["pdv_id"] = pd.to_numeric(df["pdv_id"], errors="coerce")
    if "produto_id" in df.columns:
        df["produto_id"] = pd.to_numeric(df["produto_id"], errors="coerce")
    if "dt_transacao" in df.columns:
        df["dt_transacao"] = pd.to_datetime(df["dt_transacao"], errors="coerce")
    if "dt_referencia" in df.columns:
        df["dt_referencia"] = pd.to_datetime(df["dt_referencia"], errors="coerce")
    if "quantity" in df.columns:
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    return df


def week_monday(d: pd.Series | pd.DatetimeIndex) -> pd.Series:
    """Converte datas em início da semana (segunda-feira)."""
    return pd.to_datetime(d).dt.to_period("W-MON").dt.start_time


def complete_weekly_panel(df: pd.DataFrame, keys: list[str], week_col: str, freq: str = "W-MON") -> pd.DataFrame:
    """Completa o painel semanal por grupo sem materializar o produto cartesiano.

    Para cada combinação em `keys`, reindexa o histórico para cobrir todas as semanas
    entre o mínimo e o máximo global, retornando linhas faltantes com NA (a serem
    preenchidas posteriormente). Isso reduz o pico de memória em relação ao merge
    cartesiano anterior.
    """
    if df.empty:
        return df.copy()

    df = df.copy()
    df[week_col] = pd.to_datetime(df[week_col])
    df = df.sort_values(keys + [week_col])

    df = df.set_index(keys + [week_col])
    level_names = df.index.names
    n_key_levels = len(keys)

    def _reindex(group: pd.DataFrame) -> pd.DataFrame:
        base = group.droplevel(list(range(n_key_levels)))
        if base.empty:
            return base
        local_weeks = pd.date_range(base.index.min(), base.index.max(), freq=pd.DateOffset(days=7))
        filled = base.reindex(local_weeks)
        filled.index.name = week_col
        return filled

    filled = df.groupby(level=keys, sort=False, group_keys=True).apply(_reindex)
    filled.index.set_names(level_names, inplace=True)
    return filled.reset_index()


def add_calendar_features(df: pd.DataFrame, week_col: str) -> pd.DataFrame:
    df = df.copy()
    w = pd.to_datetime(df[week_col])
    df["week"] = w.dt.isocalendar().week.astype(int)
    df["month"] = w.dt.month.astype(int)
    df["quarter"] = w.dt.quarter.astype(int)
    df["year"] = w.dt.year.astype(int)
    df["is_year_end"] = (df["week"] >= 51).astype(int)
    df["is_month_start"] = w.dt.is_month_start.astype(int)
    df["is_month_end"] = w.dt.is_month_end.astype(int)
    return df


def group_lags(df: pd.DataFrame, group_cols: list[str], target_col: str, week_col: str, lags=(1, 4, 8, 12, 26)) -> pd.DataFrame:
    df = df.sort_values(group_cols + [week_col]).copy()
    g = df.groupby(group_cols, observed=True)[target_col]
    for L in lags:
        df[f"lag_{target_col}_{L}"] = g.shift(L)
    df[f"ma_{target_col}_4"] = g.shift(1).rolling(4, min_periods=1).mean()
    df[f"ma_{target_col}_12"] = g.shift(1).rolling(12, min_periods=1).mean()
    return df


def future_weeks(start_week: pd.Timestamp, n: int) -> list[pd.Timestamp]:
    return [start_week + pd.to_timedelta(7 * i, unit="D") for i in range(n)]


def wmape(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray, eps: float = 1e-9) -> float:
    """Weighted MAPE = sum(|y - yhat|) / sum(|y|).

    Retorna proporção (0-1). Multiplique por 100 para %.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    num = np.abs(yt - yp).sum()
    den = np.abs(yt).sum()
    return float(num / max(den, eps))


def add_stat_features(df: pd.DataFrame, group_cols: list[str], target_col: str, week_col: str) -> pd.DataFrame:
    """Adiciona estatísticas adicionais: stds, diffs e ratios."""
    df = df.sort_values(group_cols + [week_col]).copy()
    g = df.groupby(group_cols, observed=True)[target_col]
    df[f"std_{target_col}_4"] = g.shift(1).rolling(4, min_periods=2).std()
    df[f"std_{target_col}_12"] = g.shift(1).rolling(12, min_periods=2).std()

    # diffs/ratios recentes
    df[f"diff_{target_col}_1"] = df[f"lag_{target_col}_1"] - df.get(f"lag_{target_col}_2")
    denom = (df.get(f"lag_{target_col}_2").abs() + 1e-6)
    df[f"ratio_{target_col}_12"] = df[f"lag_{target_col}_1"] / denom
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Deriva preço médio por unidade e médias móveis.

    Requer colunas: net_value, qty.
    """
    df = df.copy()
    with np.errstate(divide='ignore', invalid='ignore'):
        price = np.where(df["qty"] > 0, df["net_value"] / df["qty"], np.nan)
    df["price_u"] = price
    # médias móveis simples com forward fill limitado
    df["price_u_ma8"] = (
        df.groupby(["pdv_id", "produto_id"], observed=True)["price_u"]
          .apply(lambda s: s.shift(1).rolling(8, min_periods=1).mean())
          .reset_index(level=[0,1], drop=True)
    )
    return df


def add_activity_features(df: pd.DataFrame, group_cols: list[str], target_col: str, week_col: str) -> pd.DataFrame:
    """Adiciona features de atividade/recência por grupo.

    - weeks_since_last_sale: semanas desde a última venda (>0)
    - weeks_since_first_sale: semanas desde a primeira venda (>0)
    - active_share_{8,13,26}: fração de semanas com venda em janelas recentes (causal)
    - zero_streak: sequência de zeros até a semana anterior
    """
    df = df.sort_values(group_cols + [week_col]).copy()
    key = group_cols
    active = (df[target_col] > 0).astype(int)

    def _by_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        idx = np.arange(len(g))
        act = (g[target_col].values > 0).astype(int)
        # since last sale
        last_idx = pd.Series(np.where(act == 1, idx, np.nan)).ffill().to_numpy()
        w_since_last = idx - np.where(np.isnan(last_idx), -1, last_idx)
        w_since_last[np.isnan(last_idx)] = idx[np.isnan(last_idx)] + 1  # grande no início
        # since first sale
        first_idx = pd.Series(np.where(act == 1, idx, np.nan)).bfill().to_numpy()
        w_since_first = idx - np.where(np.isnan(first_idx), idx, first_idx)
        w_since_first[np.isnan(first_idx)] = 0
        # zero streak (até semana anterior)
        z = 0
        zs = []
        for a in act:
            zs.append(z)
            z = 0 if a == 1 else z + 1
        # active_share janelas (causal -> shift)
        s = pd.Series(act)
        g[f"active_share_8"] = s.shift(1).rolling(8, min_periods=1).mean().to_numpy()
        g[f"active_share_13"] = s.shift(1).rolling(13, min_periods=1).mean().to_numpy()
        g[f"active_share_26"] = s.shift(1).rolling(26, min_periods=1).mean().to_numpy()
        g["weeks_since_last_sale"] = w_since_last
        g["weeks_since_first_sale"] = w_since_first
        g["zero_streak"] = np.array(zs)
        return g

    df = df.groupby(key, observed=True, group_keys=False).apply(_by_group)
    return df


def add_discount_promo(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    with np.errstate(divide='ignore', invalid='ignore'):
        rate = np.where(df["gross_value"] > 0, df["discount"] / df["gross_value"], 0.0)
    df["discount_rate"] = np.clip(rate, 0, 1)
    df["promo_flag"] = (df["discount_rate"] >= 0.05).astype(int)
    return df


def add_holiday_features(df: pd.DataFrame, week_col: str, country: str = "US") -> pd.DataFrame:
    df = df.copy()
    if _hol is None:
        df["is_holiday"] = 0
        df["is_bf_cm_week"] = 0
        return df
    weeks = pd.to_datetime(df[week_col])
    years = sorted({d.year for d in weeks})
    if country.upper() == "US":
        cal = _hol.US(years=years)
    else:
        cal = _hol.CountryHoliday(country.upper(), years=years)
    # marca se a segunda-feira da semana é feriado
    df["is_holiday"] = weeks.dt.date.map(lambda d: 1 if d in cal else 0)
    # black friday/cyber monday weeks (hardcode 2022)
    bf_week_mondays = {pd.Timestamp("2022-11-21"), pd.Timestamp("2022-11-28")}
    df["is_bf_cm_week"] = weeks.isin(bf_week_mondays).astype(int)
    return df
