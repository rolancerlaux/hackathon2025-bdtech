from __future__ import annotations
import argparse
import os
import pandas as pd
import numpy as np

from .utils import (
    ensure_dir,
    consolidate_schema,
    week_monday,
    complete_weekly_panel,
    add_calendar_features,
    group_lags,
    add_stat_features,
    add_price_features,
    add_activity_features,
    add_discount_promo,
    add_holiday_features,
    wmape,
)
from .models import forecast_by_name


def prepare_weekly(input_path: str, outdir: str) -> dict:
    ensure_dir(outdir)
    print(f"> Lendo: {input_path}")
    df = pd.read_parquet(input_path)
    df = consolidate_schema(df)

    # Trata devoluções e garante alvo não-negativo
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0)
    df["quantity_devolucao"] = df["quantity"].where(df["quantity"] < 0, 0.0).abs()
    df["quantity_pos"] = df["quantity"].clip(lower=0)

    # Agregação semanal PDV x Produto
    df["week_start"] = week_monday(df["dt_transacao"])  # segunda-feira
    grouped = (
        df.groupby(["pdv_id", "produto_id", "week_start"], observed=True)
          .agg(qty=("quantity_pos", "sum"),
               qty_devolucao=("quantity_devolucao", "sum"),
               gross_value=("gross_value", "sum"),
               net_value=("net_value", "sum"),
               gross_profit=("gross_profit", "sum"),
               discount=("discount", "sum"),
               taxes=("taxes", "sum"),
               n_tx=("dt_transacao", "count"))
          .reset_index()
    )

    # Completa grade (zeros explícitos)
    weekly = complete_weekly_panel(grouped, keys=["pdv_id", "produto_id"], week_col="week_start")
    num_cols = [c for c in weekly.columns if c not in ["pdv_id", "produto_id", "week_start"]]
    for c in num_cols:
        weekly[c] = weekly[c].fillna(0)

    # Atributos de PDV (ex.: categoria_pdv) para backoff hierárquico (se existirem no consolidado)
    pdv_cols = [c for c in ["categoria_pdv", "premise", "zipcode"] if c in df.columns]
    if pdv_cols:
        pdv_attrs = df[["pdv_id", *pdv_cols]].drop_duplicates("pdv_id")
        weekly = weekly.merge(pdv_attrs, on="pdv_id", how="left")

    # Features
    weekly = add_calendar_features(weekly, week_col="week_start")
    weekly = group_lags(weekly, group_cols=["pdv_id", "produto_id"], target_col="qty", week_col="week_start", lags=(1,2,3,4,8,12,26))
    weekly = add_stat_features(weekly, group_cols=["pdv_id", "produto_id"], target_col="qty", week_col="week_start")
    weekly = add_price_features(weekly)
    weekly = add_activity_features(weekly, group_cols=["pdv_id", "produto_id"], target_col="qty", week_col="week_start")
    weekly = add_discount_promo(weekly)
    weekly = add_holiday_features(weekly, week_col="week_start", country="US")
    # Encodings causais (sem vazamento): contagens e médias históricas até a semana anterior
    weekly = weekly.sort_values(["pdv_id", "produto_id", "week_start"]).copy()
    # Por PDV
    weekly["freq_pdv"] = weekly.groupby("pdv_id", observed=True).cumcount()
    weekly["avg_qty_pdv"] = (
        weekly.groupby("pdv_id", observed=True)["qty"].apply(lambda s: s.shift(1).expanding(min_periods=1).mean())
             .reset_index(level=0, drop=True)
    )
    # Por Produto
    weekly["freq_prod"] = weekly.groupby("produto_id", observed=True).cumcount()
    weekly["avg_qty_prod"] = (
        weekly.groupby("produto_id", observed=True)["qty"].apply(lambda s: s.shift(1).expanding(min_periods=1).mean())
             .reset_index(level=0, drop=True)
    )

    # Split temporal: treino até 2022-12-04; validação = dezembro/2022
    cutoff = np.datetime64("2022-12-05")
    mask_train = weekly["week_start"].values < cutoff

    # Persistência
    path_weekly = os.path.join(outdir, "weekly.parquet")
    path_train = os.path.join(outdir, "train.parquet")
    path_valid = os.path.join(outdir, "valid.parquet")
    weekly.to_parquet(path_weekly, index=False)
    weekly.loc[mask_train].to_parquet(path_train, index=False)
    weekly.loc[~mask_train].to_parquet(path_valid, index=False)

    print("✅ Preparação concluída:")
    print(f"→ {path_weekly}\n→ {path_train}\n→ {path_valid}")
    return {"weekly": path_weekly, "train": path_train, "valid": path_valid}


def forecast_jan(weekly_path: str, out_path: str, baseline: str = "smart", alpha: float = 0.5,
                 fmt: str = "csv", only_active: bool = False, active_window: int = 13) -> str:
    ensure_dir(os.path.dirname(out_path) or ".")
    weekly = pd.read_parquet(weekly_path)
    # Garantir ordenação
    weekly = weekly.sort_values(["pdv_id", "produto_id", "week_start"]) 
    # Rodar baseline selecionado para 5 semanas futuras
    fcst = forecast_by_name(weekly, baseline=baseline, alpha=alpha, group_cols=("pdv_id", "produto_id"), week_col="week_start", target_col="qty", horizon=5)

    # Filtrar apenas semanas de jan/2023 e mapear para 1..5
    fcst["week_start"] = pd.to_datetime(fcst["week_start"])  # garantir datetime
    jan_mask = (fcst["week_start"] >= pd.Timestamp("2023-01-01")) & (fcst["week_start"] < pd.Timestamp("2023-02-01"))
    fcst = fcst[jan_mask].copy()
    # Ordena e cria índice 1..5 por data da semana
    jan_weeks_sorted = sorted(fcst["week_start"].unique())
    week_to_idx = {w: i + 1 for i, w in enumerate(jan_weeks_sorted)}
    fcst["semana"] = fcst["week_start"].map(week_to_idx).astype(int)

    # Formato de entrega: [semana, pdv, produto, quantidade] como inteiros
    pred_col = {
        "ma4": "pred_ma4",
        "naive": "pred_naive",
        "ewma": "pred_ewma",
        "smart": "pred_smart",
    }.get(baseline.lower(), "pred_smart")

    out = fcst.rename(columns={
        "pdv_id": "pdv",
        "produto_id": "produto",
        pred_col: "quantidade",
    })[["semana", "pdv", "produto", "quantidade"]].copy()
    # Quantidade não-negativa e inteira
    out["quantidade"] = np.rint(out["quantidade"].clip(lower=0)).astype("int64")
    out["pdv"] = pd.to_numeric(out["pdv"], errors="coerce").fillna(0).astype("int64")
    out["produto"] = pd.to_numeric(out["produto"], errors="coerce").fillna(0).astype("int64")
    out["semana"] = pd.to_numeric(out["semana"], errors="coerce").fillna(0).astype("int64")

    # Opcional: reduzir quantidade de pares para diminuir tamanho
    if only_active:
        jan_start = pd.Timestamp("2023-01-01")
        window_start = jan_start - pd.to_timedelta(active_window * 7, unit="D")
        recent_pairs = weekly[(weekly["week_start"] >= window_start) & (weekly["week_start"] < jan_start) & (weekly["qty"] > 0)][["pdv_id", "produto_id"]].drop_duplicates()
        recent_pairs = recent_pairs.rename(columns={"pdv_id": "pdv", "produto_id": "produto"})
        out = out.merge(recent_pairs.assign(_keep=1), on=["pdv", "produto"], how="inner").drop(columns=["_keep"])  # só pares ativos

    # Persistência no formato desejado
    fmt = fmt.lower()
    if fmt == "csv":
        out.to_csv(out_path, sep=";", encoding="utf-8", index=False, header=False)
    elif fmt == "parquet":
        out.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")
    else:
        raise ValueError("Formato inválido. Use 'csv' ou 'parquet'.")
    print(f"✅ Forecast salvo em: {out_path}")
    return out_path


def evaluate_valid(weekly_path: str, out_csv: str, cutoff: str = "2022-12-05", baseline: str = "smart", alpha: float = 0.5, restrict_lags: bool = False) -> dict:
    """Avalia baseline MA(4) na janela de validação (dez/2022).

    - Usa histórico até `cutoff` (exclusivo) para gerar previsões recursivas.
    - Compara com o `qty` observado nas semanas >= cutoff e < 2023-01-01.
    - Calcula WMAPE global e salva CSV com previsões vs. verdade.
    """
    ensure_dir(os.path.dirname(out_csv) or ".")
    weekly = pd.read_parquet(weekly_path)
    weekly = weekly.sort_values(["pdv_id", "produto_id", "week_start"]).copy()

    cutoff_ts = pd.Timestamp(cutoff)
    end_val = pd.Timestamp("2023-01-01")

    hist = weekly[weekly["week_start"] < cutoff_ts]
    valid = weekly[(weekly["week_start"] >= cutoff_ts) & (weekly["week_start"] < end_val)]

    # horizonte = nº de semanas únicas na validação
    horizon = valid["week_start"].nunique()
    if horizon == 0:
        raise ValueError("Não há semanas de validação após o cutoff informado.")

    preds = forecast_by_name(hist, baseline=baseline, alpha=alpha, group_cols=("pdv_id", "produto_id"), week_col="week_start", target_col="qty", horizon=horizon)

    # manter apenas semanas na janela de validação
    preds = preds[(preds["week_start"] >= cutoff_ts) & (preds["week_start"] < end_val)].copy()

    # Descobre coluna de previsão conforme baseline
    pred_col = {
        "ma4": "pred_ma4",
        "naive": "pred_naive",
        "ewma": "pred_ewma",
        "smart": "pred_smart",
    }.get(baseline.lower(), "pred_smart")

    df_eval = valid.merge(
        preds.rename(columns={pred_col: "yhat"}),
        on=["pdv_id", "produto_id", "week_start"],
        how="left",
    )
    df_eval["abs_err"] = (df_eval["qty"] - df_eval["yhat"]).abs()

    # Opcional: restringe a avaliação a linhas com lags disponíveis (comparável ao ML)
    if restrict_lags:
        lag_cols = [c for c in df_eval.columns if c.startswith("lag_qty_")]
        if lag_cols:
            df_eval = df_eval.dropna(subset=lag_cols)

    score_all = wmape(df_eval["qty"], df_eval["yhat"])  # 0-1
    pos_mask = df_eval["qty"] > 0
    score_pos = wmape(df_eval.loc[pos_mask, "qty"], df_eval.loc[pos_mask, "yhat"]) if pos_mask.any() else float("nan")

    # Salva CSV: semana (segunda), pdv, produto, y_true, y_pred, abs_err
    out = df_eval.rename(columns={
        "week_start": "semana",
        "pdv_id": "pdv",
        "produto_id": "produto",
        "qty": "y_true",
        "yhat": "y_pred",
    })[["semana", "pdv", "produto", "y_true", "y_pred", "abs_err"]].copy()
    # Salva com ; e UTF-8 (arquivo de diagnóstico com header)
    out.to_csv(out_csv, sep=";", encoding="utf-8", index=False)

    print("✅ Avaliação concluída (dez/2022):")
    print(f"Semanas avaliadas: {horizon}")
    print(f"Linhas avaliadas: {len(out):,}")
    # WMAPE com vírgula decimal (0,575323)
    print(f"Baseline: {baseline}")
    score_all_str = f"{score_all:.6f}".replace(".", ",")
    score_pos_str = (f"{score_pos:.6f}".replace(".", ",") if not np.isnan(score_pos) else "NA")
    print(f"WMAPE_all: {score_all_str}")
    print(f"WMAPE_pos: {score_pos_str}")
    print(f"→ {out_csv}")
    return {"wmape_all": score_all, "wmape_pos": score_pos, "rows": len(out), "horizon": horizon, "csv": out_csv}


def train_ml(weekly_path: str, model_out: str, cutoff: str = "2022-12-05", two_stage: bool = False, tau: float = 0.5) -> dict:
    """Treina LightGBM Regressor em features de lags/calendário e avalia na validação com WMAPE."""
    from lightgbm import LGBMRegressor
    try:
        from lightgbm.callback import early_stopping as lgb_early_stopping, log_evaluation as lgb_log_evaluation
    except Exception:
        lgb_early_stopping = None
        lgb_log_evaluation = None
    weekly = pd.read_parquet(weekly_path)
    weekly = weekly.sort_values(["pdv_id", "produto_id", "week_start"]).copy()

    cutoff_ts = pd.Timestamp(cutoff)
    end_val = pd.Timestamp("2023-01-01")

    train = weekly[weekly["week_start"] < cutoff_ts].copy()
    valid = weekly[(weekly["week_start"] >= cutoff_ts) & (weekly["week_start"] < end_val)].copy()

    # Seleção de features
    feat_cols = [
        "lag_qty_1","lag_qty_2","lag_qty_3","lag_qty_4","lag_qty_8","lag_qty_12","lag_qty_26",
        "ma_qty_4","ma_qty_12",
        "std_qty_4","std_qty_12",
        "diff_qty_1","ratio_qty_12",
        "n_tx","week","month","quarter","is_year_end","is_month_start","is_month_end",
        "price_u_ma8",
        "freq_pdv","avg_qty_pdv","freq_prod","avg_qty_prod",
        "weeks_since_last_sale","weeks_since_first_sale","zero_streak",
        "active_share_8","active_share_13","active_share_26",
        "discount_rate","promo_flag","is_holiday","is_bf_cm_week"
    ]

    # Remove linhas sem lags suficientes
    def _clean(df):
        return df.dropna(subset=[c for c in feat_cols if c in df.columns]).copy()

    train = _clean(train)
    valid = _clean(valid)

    X_tr = train[feat_cols]
    y_tr = train["qty"].astype(float)
    X_va = valid[feat_cols]
    y_va = valid["qty"].astype(float)

    params = dict(
        n_estimators=5000,
        learning_rate=0.03,
        max_depth=-1,
        num_leaves=63,
        min_child_samples=40,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="poisson",
        reg_lambda=5.0,
        reg_alpha=0.5,
        random_state=42,
        n_jobs=-1,
    )
    # early stopping para reduzir sobreajuste (via callbacks p/ compatibilidade)
    callbacks = []
    if lgb_early_stopping is not None:
        callbacks.append(lgb_early_stopping(stopping_rounds=200))
    if lgb_log_evaluation is not None:
        callbacks.append(lgb_log_evaluation(50))
    if not two_stage:
        model = LGBMRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="l1", callbacks=callbacks)
        y_hat = np.maximum(0.0, model.predict(X_va))
        score = wmape(y_va, y_hat)
        score_str = f"{score:.6f}".replace(".", ",")
        ensure_dir(os.path.dirname(model_out) or ".")
        import pickle
        with open(model_out, "wb") as f:
            pickle.dump({"model": model, "features": feat_cols, "cutoff": cutoff, "type": "regressor"}, f)
        print("✅ Treino concluído (LightGBM):")
        print(f"Linhas treino: {len(X_tr):,} | validação: {len(X_va):,}")
        print(f"WMAPE validação: {score_str}")
        print(f"→ {model_out}")
        return {"wmape": score, "rows_train": len(X_tr), "rows_valid": len(X_va), "model": model_out}
    else:
        from lightgbm import LGBMClassifier
        y_bin_tr = (y_tr > 0).astype(int)
        y_bin_va = (y_va > 0).astype(int)
        clf = LGBMClassifier(
            n_estimators=3000,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=5.0,
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_tr, y_bin_tr, eval_set=[(X_va, y_bin_va)], eval_metric="binary_logloss", callbacks=callbacks)
        reg = LGBMRegressor(**params)
        pos_mask_tr = y_tr > 0
        reg.fit(X_tr[pos_mask_tr], y_tr[pos_mask_tr], eval_set=[(X_va[y_va > 0], y_va[y_va > 0])], eval_metric="l1", callbacks=callbacks)
        proba = clf.predict_proba(X_va)[:, 1]
        qty_hat = np.maximum(0.0, reg.predict(X_va))
        y_hat = np.where(proba >= tau, qty_hat, 0.0)
        score = wmape(y_va, y_hat)
        score_str = f"{score:.6f}".replace(".", ",")
        ensure_dir(os.path.dirname(model_out) or ".")
        import pickle
        with open(model_out, "wb") as f:
            pickle.dump({"clf": clf, "reg": reg, "features": feat_cols, "tau": tau, "cutoff": cutoff, "type": "two_stage"}, f)
        print("✅ Treino concluído (Two-Stage LightGBM):")
        print(f"Linhas treino: {len(X_tr):,} | validação: {len(X_va):,}")
        print(f"WMAPE validação: {score_str}")
        print(f"→ {model_out}")
        return {"wmape": score, "rows_train": len(X_tr), "rows_valid": len(X_va), "model": model_out}


def main():
    parser = argparse.ArgumentParser(description="Pipeline: preparar dados e gerar baseline para jan/2023")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser("prepare", help="Agrega semanal e cria features")
    p_prep.add_argument("--input", required=True, help="Caminho do dataset consolidado (parquet)")
    p_prep.add_argument("--outdir", default="data/processed", help="Diretório de saída")

    p_fc = sub.add_parser("forecast", help="Gera baseline para jan/2023")
    p_fc.add_argument("--weekly", required=True, help="Caminho para weekly.parquet gerado no prepare")
    p_fc.add_argument("--out", default="outputs/forecast_jan_2023.csv", help="Caminho do arquivo de saída (.csv ou .parquet)")
    p_fc.add_argument("--baseline", default="smart", choices=["ma4", "naive", "ewma", "smart"], help="Tipo de baseline")
    p_fc.add_argument("--alpha", type=float, default=0.5, help="Alpha para EWMA (0-1)")
    p_fc.add_argument("--format", dest="fmt", default="csv", choices=["csv", "parquet"], help="Formato de saída")
    p_fc.add_argument("--only-active", action="store_true", help="Incluir apenas pares com atividade recente para reduzir tamanho")
    p_fc.add_argument("--active-window", type=int, default=13, help="Janela (semanas) para considerar atividade recente")

    p_eval = sub.add_parser("evaluate", help="Avalia baseline na validação com WMAPE")
    p_eval.add_argument("--weekly", required=True, help="Caminho para weekly.parquet gerado no prepare")
    p_eval.add_argument("--out", default="outputs/eval_valid_dec2022.csv", help="CSV de previsões vs. verdade para validação")
    p_eval.add_argument("--cutoff", default="2022-12-05", help="Data de corte (segunda-feira) que separa treino/validação")
    p_eval.add_argument("--baseline", default="smart", choices=["ma4", "naive", "ewma", "smart"], help="Tipo de baseline")
    p_eval.add_argument("--alpha", type=float, default=0.5, help="Alpha para EWMA (0-1)")
    p_eval.add_argument("--restrict-lags", action="store_true", help="Avaliar apenas linhas com lags disponíveis (comparável ao ML)")

    p_train = sub.add_parser("train", help="Treina LightGBM e avalia na validação")
    p_train.add_argument("--weekly", required=True, help="Caminho para weekly.parquet gerado no prepare")
    p_train.add_argument("--out", default="models/lgbm.pkl", help="Caminho do arquivo de modelo (pickle)")
    p_train.add_argument("--cutoff", default="2022-12-05", help="Data de corte (segunda-feira) que separa treino/validação")
    p_train.add_argument("--two-stage", action="store_true", help="Treinar pipeline em 2 estágios (classificador + regressor)")
    p_train.add_argument("--tau", type=float, default=0.5, help="Limiar de probabilidade para prever >0 no modelo de 2 estágios")

    args = parser.parse_args()
    if args.cmd == "prepare":
        prepare_weekly(args.input, args.outdir)
    elif args.cmd == "forecast":
        forecast_jan(args.weekly, args.out, args.baseline, args.alpha, args.fmt, args.only_active, args.active_window)
    elif args.cmd == "evaluate":
        evaluate_valid(args.weekly, args.out, args.cutoff, args.baseline, args.alpha, args.restrict_lags)
    elif args.cmd == "train":
        train_ml(args.weekly, args.out, args.cutoff, args.two_stage, args.tau)


if __name__ == "__main__":
    main()
