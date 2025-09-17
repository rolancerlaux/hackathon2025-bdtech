# -*- coding: utf-8 -*-
"""
Pipeline de preparação de dados para previsão de vendas por PDV x Produto (base 2022).

Entradas esperadas:
- Arquivo 1 (cadastro de PDVs): colunas ['pdv', 'premise', 'categoria_pdv', 'zipcode']
- Arquivo 2 (transações 2022): colunas ['internal_store_id','internal_product_id','distributor_id','transaction_date','reference_date','quantity','gross_value','net_value','gross_profit','discount','taxes']
- Arquivo 3 (cadastro de produtos): colunas ['produto','categoria','descricao','tipos','label','subcategoria','marca','fabricante']

Saídas:
- dataset_semanal.parquet: granularidade semanal por (pdv, produto), com features de histórico e atributos de PDV/produto.
- train.parquet / valid.parquet: split temporal (validação = últimas 4 semanas de 2022).

Observações/assunções:
- Mapeamento de chaves: internal_store_id ↔ pdv; internal_product_id ↔ produto (ajuste se houver chave distinta).
- Usamos transaction_date como data de ocorrência. reference_date pode ser usado para alinhamento semanal se desejado.
- Quantidades negativas são tratadas como devoluções. Mantemos a coluna quantity_devolucao e usamos quantity_pos para previsão de sell-out.
- Frequência semanal inicia segunda-feira (W-MON).
"""

from __future__ import annotations
import os
import pandas as pd
import numpy as np, gc, os
from typing import Tuple

# ===============================
# Configuração de caminhos (ajuste)
# ===============================
DATA_DIR = "data/raw/"
PATH_PDV = f"{DATA_DIR}cadastro_pdvs.parquet"
PATH_TX  = f"{DATA_DIR}transacoes_2022.parquet"
PATH_PRD = f"{DATA_DIR}cadastro_produtos.parquet"
OUTPUT_DIR = "data/processed/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================
# Utilitários
# ===============================

def _standardize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def _parse_dates(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def _validate_date_range(df: pd.DataFrame, date_col: str, start: str = "2022-01-01", end: str = "2022-12-31"):
    if date_col not in df.columns:
        return
    min_d, max_d = df[date_col].min(), df[date_col].max()
    print(f"Intervalo encontrado em {date_col}: {min_d} → {max_d}")
    # Opcional: filtrar estranhos fora de 2022
    mask_year = (df[date_col] >= pd.to_datetime(start)) & (df[date_col] <= pd.to_datetime(end))
    outliers = (~mask_year).sum()
    if outliers > 0:
        print(f"⚠️ {outliers} registros fora de 2022 em {date_col} (serão filtrados)")
    return df.loc[mask_year].copy()


def _complete_panel(df: pd.DataFrame, keys: list[str], date_col: str) -> pd.DataFrame:
    """Cria grade completa de (keys x semanas) para evitar perda de zeros."""
    df = df.copy()
    all_keys = df[keys].drop_duplicates()
    all_weeks = pd.DataFrame({date_col: sorted(df[date_col].unique())})
    all_weeks["tmp_key"] = 1
    all_keys["tmp_key"] = 1
    grid = all_keys.merge(all_weeks, on="tmp_key", how="outer").drop(columns="tmp_key")
    out = grid.merge(df, on=keys + [date_col], how="left")
    return out


def _add_calendar_features(df: pd.DataFrame, week_col: str) -> pd.DataFrame:
    df = df.copy()
    w = pd.to_datetime(df[week_col])
    df["week"] = w.dt.isocalendar().week.astype(int)
    df["month"] = w.dt.month.astype(int)
    df["quarter"] = w.dt.quarter.astype(int)
    df["year"] = w.dt.year.astype(int)
    # Placeholder para feriados: atribuir via calendário externo, se disponível
    df["is_year_end"] = (df["week"] >= 51).astype(int)
    df["is_month_start"] = (w.dt.is_month_start).astype(int)
    df["is_month_end"] = (w.dt.is_month_end).astype(int)
    return df


def _group_lags(df: pd.DataFrame, group_cols: list[str], target_col: str, lags=(1,4,12)) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(group_cols + ["week_start"])  # week_start já deve existir
    g = df.groupby(group_cols, observed=True)[target_col]
    for L in lags:
        df[f"lag_{target_col}_{L}"] = g.shift(L)
    # médias móveis (janela alin. ao passado)
    df[f"ma_{target_col}_4"] = g.shift(1).rolling(4, min_periods=1).mean()
    df[f"ma_{target_col}_12"] = g.shift(1).rolling(12, min_periods=1).mean()
    return df


# ===============================
# Carregamento
# ===============================
print("> Lendo arquivos...")
pdv = _standardize_cols(pd.read_parquet(PATH_PDV))
tx  = _standardize_cols(pd.read_parquet(PATH_TX))
prd = _standardize_cols(pd.read_parquet(PATH_PRD))

# Renomear chaves para padronizar
pdv = pdv.rename(columns={"pdv": "pdv_id"})
tx = tx.rename(columns={
    "internal_store_id": "pdv_id",
    "internal_product_id": "produto_id",
    "transaction_date": "dt_transacao",
    "reference_date": "dt_referencia"
})
prd = prd.rename(columns={"produto": "produto_id"})

# Tipagem de colunas principais
int_cols = ["pdv_id", "produto_id", "distributor_id"]
for c in int_cols:
    if c in tx.columns:
        tx[c] = pd.to_numeric(tx[c], errors="coerce")
    if c == "pdv_id" and c in pdv.columns:
        pdv[c] = pd.to_numeric(pdv[c], errors="coerce")
    if c == "produto_id" and c in prd.columns:
        prd[c] = pd.to_numeric(prd[c], errors="coerce")

# Datas
_tx_dates = ["dt_transacao", "dt_referencia"]
tx = _parse_dates(tx, _tx_dates)

# Filtra ano-base
print("> Validando intervalo de datas...")
tx = _validate_date_range(tx, "dt_transacao")

# Remoção de duplicidades óbvias
before = len(tx)
tx = tx.drop_duplicates(subset=["pdv_id", "produto_id", "dt_transacao", "gross_value", "quantity"])  # heurística
print(f"> Removidos {before - len(tx)} duplicados (heurística)")

# Tratamento de quantidades negativas (devolução)
tx["quantity"] = pd.to_numeric(tx["quantity"], errors="coerce")
tx["quantity_devolucao"] = tx["quantity"].where(tx["quantity"] < 0, 0.0).abs()
tx["quantity_pos"] = tx["quantity"].clip(lower=0)

# ===============================
# Agregação semanal
# ===============================
print("> Agregando por semana PDV x Produto...")
# Semana iniciando na segunda-feira
week = tx["dt_transacao"].dt.to_period("W-MON").dt.start_time

df_w = (
    tx.assign(week_start=week)
      .groupby(["pdv_id", "produto_id", "week_start"], observed=True)
      .agg(
          qty=("quantity_pos", "sum"),
          qty_devolucao=("quantity_devolucao", "sum"),
          gross_value=("gross_value", "sum"),
          net_value=("net_value", "sum"),
          gross_profit=("gross_profit", "sum"),
          discount=("discount", "sum"),
          taxes=("taxes", "sum"),
          n_tx=("dt_transacao", "count")
      )
      .reset_index()
)

# Completar grade para semanas sem venda
print("> Completando grade (zeros explícitos)...")
df_w = _complete_panel(df_w, keys=["pdv_id", "produto_id"], date_col="week_start")

# Zeros em métricas numéricas onde NA
num_cols = [c for c in df_w.columns if c not in ["pdv_id", "produto_id", "week_start"]]
for c in num_cols:
    df_w[c] = df_w[c].fillna(0)

# ===============================
# Join com cadastros
# ===============================
print("> Enriquecendo com atributos de PDV e Produto...")
df = df_w.merge(pdv, on="pdv_id", how="left")
df = df.merge(prd, on="produto_id", how="left")

# ===============================
# Features de calendário e lags
# ===============================
print("> Criando features de calendário e histórico...")
df = _add_calendar_features(df, week_col="week_start")
df = _group_lags(df, group_cols=["pdv_id", "produto_id"], target_col="qty", lags=(1,4,8,12,26))

# Dummies (one-hot) para variáveis categóricas leves
#cat_cols = [c for c in ["premise", "categoria_pdv", "zipcode", "categoria", "subcategoria", "marca", "fabricante", "label", "tipos"] if c in df.columns]
#df = pd.get_dummies(df, columns=cat_cols, dummy_na=True)

# ===============================
# Split temporal (validação = últimas 4 semanas de 2022)
# ===============================
print("> Split temporal train/valid...")
cutoff = np.datetime64("2022-12-05")  # segunda-feira 2022-12-05 (4 semanas finais: 05, 12, 19, 26)
mask = df["week_start"].values < cutoff

# ===============================
# Export
# ===============================
print("> Salvando saídas...")
path_all = os.path.join(OUTPUT_DIR, "dataset_semanal.parquet")
path_train = os.path.join(OUTPUT_DIR, "train.parquet")
path_valid = os.path.join(OUTPUT_DIR, "valid.parquet")

df.to_parquet(path_all, index=False)
df.loc[mask].to_parquet(path_train, index=False, engine="pyarrow")
df.loc[~mask].to_parquet(path_valid, index=False, engine="pyarrow")

del mask
gc.collect()


print("✅ Concluído.")
print(f"→ {path_all}\n→ {path_train}\n→ {path_valid}")

# ===============================
# Baseline simples: previsão = MA4
# ===============================
#if not valid.empty:
#    # baseline para servir de referência na modelagem posterior
#    base_cols = ["pdv_id", "produto_id", "week_start", "qty", "ma_qty_4"]
#    cols_exist = [c for c in base_cols if c in df.columns]
#    baseline = df[cols_exist].copy()
#    baseline = baseline[baseline["week_start"] >= cutoff]
#    baseline = baseline.rename(columns={"ma_qty_4": "pred_baseline_ma4"})
#    baseline_path = os.path.join(OUTPUT_DIR, "baseline_validacao.parquet")
#    baseline.to_parquet(baseline_path, index=False)
#    print(f"→ {baseline_path}")
