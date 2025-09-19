from __future__ import annotations
import pandas as pd
import numpy as np


def ma4_recursive_forecast(weekly_df: pd.DataFrame,
                           group_cols=("pdv_id", "produto_id"),
                           week_col="week_start",
                           target_col="qty",
                           horizon=5) -> pd.DataFrame:
    """Faz previsão recursiva usando MA(4) por grupo.

    - Usa as últimas até 4 observações de `target_col` por grupo.
    - Para passos > 1, utiliza previsões anteriores na janela.
    - Não impõe não-negatividade, mas pode-se aplicar clip(0) depois.
    """
    df = weekly_df[[*group_cols, week_col, target_col]].copy()
    df = df.sort_values(list(group_cols) + [week_col])

    results = []
    for keys, g in df.groupby(list(group_cols), observed=True):
        g = g.sort_values(week_col).reset_index(drop=True)
        history = g[target_col].astype(float).tolist()
        last_week = pd.to_datetime(g[week_col].iloc[-1])
        # gerar semanas futuras
        weeks = [last_week + pd.to_timedelta(7 * (i + 1), unit="D") for i in range(horizon)]
        preds = []
        for _ in range(horizon):
            window = history[-4:] if len(history) >= 1 else []
            if len(window) == 0:
                yhat = 0.0
            else:
                yhat = float(np.mean(window))
            preds.append(max(0.0, yhat))
            history.append(yhat)
        out = pd.DataFrame({
            week_col: weeks,
            target_col: preds,
        })
        for c, v in zip(group_cols, keys if isinstance(keys, tuple) else (keys,)):
            out[c] = v
        results.append(out)

    fcst = pd.concat(results, ignore_index=True)
    cols = [*group_cols, week_col, target_col]
    return fcst[cols].rename(columns={target_col: "pred_ma4"})


def naive_last_recursive_forecast(weekly_df: pd.DataFrame,
                                  group_cols=("pdv_id", "produto_id"),
                                  week_col="week_start",
                                  target_col="qty",
                                  use_last_nonzero: bool = True,
                                  horizon: int = 5) -> pd.DataFrame:
    df = weekly_df[[*group_cols, week_col, target_col, "n_tx"]].copy()
    df = df.sort_values(list(group_cols) + [week_col])

    results = []
    for keys, g in df.groupby(list(group_cols), observed=True):
        g = g.sort_values(week_col)
        last_week = pd.to_datetime(g[week_col].iloc[-1])
        weeks = [last_week + pd.to_timedelta(7 * (i + 1), unit="D") for i in range(horizon)]

        if use_last_nonzero:
            obs = g[g.get("n_tx", 1) > 0]
            if len(obs) > 0:
                last_val = float(obs[target_col].iloc[-1])
            else:
                last_val = float(g[target_col].iloc[-1]) if len(g) else 0.0
        else:
            last_val = float(g[target_col].iloc[-1]) if len(g) else 0.0

        preds = [max(0.0, last_val)] * horizon
        out = pd.DataFrame({week_col: weeks, target_col: preds})
        for c, v in zip(group_cols, keys if isinstance(keys, tuple) else (keys,)):
            out[c] = v
        results.append(out)

    fcst = pd.concat(results, ignore_index=True)
    cols = [*group_cols, week_col, target_col]
    return fcst[cols].rename(columns={target_col: "pred_naive"})


def ewma_forecast(weekly_df: pd.DataFrame,
                  group_cols=("pdv_id", "produto_id"),
                  week_col="week_start",
                  target_col="qty",
                  alpha: float = 0.5,
                  use_observed_only: bool = True,
                  horizon: int = 5) -> pd.DataFrame:
    df = weekly_df[[*group_cols, week_col, target_col, "n_tx"]].copy()
    df = df.sort_values(list(group_cols) + [week_col])

    results = []
    for keys, g in df.groupby(list(group_cols), observed=True):
        g = g.sort_values(week_col).reset_index(drop=True)
        if use_observed_only and "n_tx" in g.columns:
            series = g.loc[g["n_tx"] > 0, target_col].astype(float).tolist()
        else:
            series = g[target_col].astype(float).tolist()

        if len(series) == 0:
            s = 0.0
        else:
            s = series[0]
            for y in series[1:]:
                s = alpha * y + (1 - alpha) * s
        last_week = pd.to_datetime(g[week_col].iloc[-1])
        weeks = [last_week + pd.to_timedelta(7 * (i + 1), unit="D") for i in range(horizon)]
        preds = [max(0.0, float(s))] * horizon
        out = pd.DataFrame({week_col: weeks, target_col: preds})
        for c, v in zip(group_cols, keys if isinstance(keys, tuple) else (keys,)):
            out[c] = v
        results.append(out)

    fcst = pd.concat(results, ignore_index=True)
    cols = [*group_cols, week_col, target_col]
    return fcst[cols].rename(columns={target_col: "pred_ewma"})


def smart_backoff_forecast(weekly_df: pd.DataFrame,
                           group_cols=("pdv_id", "produto_id"),
                           week_col="week_start",
                           target_col="qty",
                           horizon: int = 5,
                           inactivity_window: int = 12) -> pd.DataFrame:
    """Baseline robusto com backoff para cold-start/esparsidade.

    Estratégia por grupo:
    1) Se há ≥4 semanas observadas (n_tx>0), usa média das últimas 4 observadas.
    2) Senão, se há ≥1 observada, usa blend: max(última observada, 0.7*media observada do grupo).
    3) Senão, cai para média do produto (em todos os PDVs, observados);
       se indisponível, média global observada.
    Repete o valor previsto para o horizonte (constante por simplicidade).
    """
    df = weekly_df[[*group_cols, week_col, target_col, "n_tx"]].copy()
    df = df.sort_values(list(group_cols) + [week_col])

    # Médias por produto e global considerando apenas semanas com vendas observadas
    obs_mask = df["n_tx"] > 0 if "n_tx" in df.columns else pd.Series(True, index=df.index)
    prod_means = df.loc[obs_mask].groupby("produto_id", observed=True)[target_col].mean().to_dict()
    # backoff hierárquico: produto x categoria_pdv
    prod_cat_means = None
    if "categoria_pdv" in df.columns:
        prod_cat_means = df.loc[obs_mask].groupby(["produto_id", "categoria_pdv"], observed=True)[target_col].mean().to_dict()
    # mapeia pdv -> categoria
    pdv_to_cat = None
    if "categoria_pdv" in df.columns:
        pdv_to_cat = df.dropna(subset=["pdv_id"]).drop_duplicates("pdv_id")[
            ["pdv_id", "categoria_pdv"]
        ].set_index("pdv_id").to_dict().get("categoria_pdv", {})
    global_mean = float(df.loc[obs_mask, target_col].mean() or 0.0)

    results = []
    for keys, g in df.groupby(list(group_cols), observed=True):
        g = g.sort_values(week_col)
        last_week = pd.to_datetime(g[week_col].iloc[-1])
        weeks = [last_week + pd.to_timedelta(7 * (i + 1), unit="D") for i in range(horizon)]

        g_obs = g[g.get("n_tx", 1) > 0]

        # Regra de inatividade recente: se nas últimas K semanas não houve venda, prever 0
        recent = g.tail(inactivity_window)
        recent_activity = False
        if "n_tx" in recent.columns:
            recent_activity = bool((recent["n_tx"] > 0).any())
        else:
            recent_activity = bool((recent[target_col] > 0).any())
        if not recent_activity:
            yhat = 0.0
        else:
            if len(g_obs) >= 4:
                yhat = float(g_obs[target_col].tail(4).mean())
            elif len(g_obs) >= 1:
                last_obs = float(g_obs[target_col].iloc[-1])
                mean_obs = float(g_obs[target_col].mean())
                yhat = max(last_obs, 0.7 * mean_obs)
            else:
                # backoff hierárquico: produto x categoria_pdv -> produto -> global
                produto_id = keys[1] if isinstance(keys, tuple) else None
                yhat = None
                if prod_cat_means is not None and pdv_to_cat is not None:
                    cat = pdv_to_cat.get(keys[0])
                    if cat is not None:
                        yhat = prod_cat_means.get((produto_id, cat))
                if yhat is None:
                    yhat = prod_means.get(produto_id, global_mean)

        yhat = max(0.0, yhat)
        out = pd.DataFrame({week_col: weeks, target_col: [yhat] * horizon})
        for c, v in zip(group_cols, keys if isinstance(keys, tuple) else (keys,)):
            out[c] = v
        results.append(out)

    fcst = pd.concat(results, ignore_index=True)
    cols = [*group_cols, week_col, target_col]
    return fcst[cols].rename(columns={target_col: "pred_smart"})


def forecast_by_name(weekly_df: pd.DataFrame,
                     baseline: str = "ma4",
                     horizon: int = 5,
                     alpha: float = 0.5,
                     group_cols=("pdv_id", "produto_id"),
                     week_col="week_start",
                     target_col="qty") -> pd.DataFrame:
    baseline = baseline.lower()
    if baseline == "ma4":
        return ma4_recursive_forecast(weekly_df, group_cols, week_col, target_col, horizon)
    if baseline == "naive":
        return naive_last_recursive_forecast(weekly_df, group_cols, week_col, target_col, True, horizon)
    if baseline == "ewma":
        return ewma_forecast(weekly_df, group_cols, week_col, target_col, alpha, True, horizon)
    if baseline == "smart":
        return smart_backoff_forecast(weekly_df, group_cols, week_col, target_col, horizon)
    raise ValueError(f"Baseline desconhecido: {baseline}")
