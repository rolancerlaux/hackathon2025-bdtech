# 🛒 Hackathon Forecast Big Data 2025  

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)  
[![Made with Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange.svg)](https://jupyter.org/)  

Este repositório contém a solução desenvolvida para o **Desafio Técnico – Hackathon Forecast Big Data 2025**, que tem como objetivo prever a **quantidade semanal de vendas por PDV (Ponto de Venda) e SKU** para apoiar o varejo na **reposição de produtos**.  

## 📌 Desafio  

- Desenvolver um **modelo de previsão de vendas (forecast)**.  
- Prever as vendas para **5 semanas de janeiro/2023**.  
- Base de treino: **histórico de 2022**.  
- Dados disponíveis:  
  - **Transações:** Data, PDV, Produto, Quantidade, Faturamento.  
  - **Cadastro de produtos:** Produto, Categoria, Descrição, + atributos.  
  - **Cadastro de PDVs:** PDV, On/Off Prem, Categoria (c-store, g-store, liquor etc.), Zipcode.  
- Dados de teste (**jan/2023**) não foram disponibilizados.  

## 📂 Estrutura do repositório  

📦 hackathon-forecast-bigdata
 ```bash
 ├── data/
 │   ├── dataset_consolidado.parquet        # dataset de entrada (ajuste nome conforme arquivo disponível)
 │   └── processed/                         # saídas geradas pelo pipeline (weekly/train/valid)
 ├── notebooks/
 │   ├── 01_eda.ipynb # Análise exploratória (EDA)
 │   ├── 02_preprocessing.ipynb # Limpeza e transformação dos dados
 │   ├── 03_modeling.ipynb # Treinamento e avaliação de modelos
 │   ├── 04_forecast.ipynb # Previsões finais
 ├── src/
 │   ├── data_preprocessing.py # Funções de tratamento de dados
 │   ├── feature_engineering.py # Criação de features
 │   ├── models.py # Implementação e treinamento de modelos
 │   └── utils.py # Funções auxiliares
 ├── requirements.txt # Dependências do projeto
 └── README.md # Documentação
 ```

## ⚙️ Tecnologias utilizadas  

- **Linguagem:** Python 3.9+  
- **Bibliotecas principais:**  
  - Pandas, NumPy (manipulação de dados)  
  - Matplotlib, Seaborn (visualização)  
  - Scikit-learn (modelos base e métricas)  
  - Statsmodels / Prophet / XGBoost / LightGBM (modelagem de séries temporais e regressão)  

## 🚀 Como executar  

1. Clone este repositório:

   ```bash
   git clone https://github.com/seu-usuario/hackathon-forecast-bigdata.git
   cd hackathon-forecast-bigdata
   ```

2. Crie e ative um ambiente virtual:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/Mac
   .venv/Scripts/activate      # Windows
   ```

3. Instale as dependências

   ```bash
   pip install -r requirements.txt
   ```

4. Execute os notebooks na pasta notebooks/ para reproduzir a solução.

### Pipeline CLI (dados → modelo → previsões)

Após instalar as dependências, execute o pipeline completo com os scripts em `src/`:

```bash
python -m src.pipeline prepare --input data/dataset_consolidado.parquet --outdir data/processed

python -m src.pipeline train \
  --weekly data/processed/weekly.parquet \
  --out models/lgbm_twostage_recencia.pkl \
  --history-weeks 52 \
  --decay-weeks 26 \
  --two-stage \
  --tau 0.35

python -m src.pipeline evaluate --weekly data/processed/weekly.parquet --out outputs/eval_valid_dec2022.csv
source .venv/bin/activate
source .venv/bin/activate
python -m src.pipeline forecast-ml \
  --weekly data/processed/weekly.parquet \
  --model models/lgbm_twostage_recencia.pkl \
  --out outputs/forecast_jan_ml_active3.parquet \
  --format parquet \
  --only-active \
  --active-window 3
```

- `prepare`: agrega o consolidado em painel semanal, cria features e salva `weekly/train/valid` em `data/processed/`. Ajuste o nome do arquivo usado em `--input` se estiver diferente (ex.: `data/dataset_consolidado.parquet`).
- `train`: treina o LightGBM (ou versão two-stage) e salva o modelo em `models/`.
- `evaluate`: calcula WMAPE na validação de dez/2022 para a baseline selecionada.
- `forecast`: gera as previsões para as 5 semanas de janeiro/2023 em formato Parquet (`--format parquet`).

## 📊 Métricas de avaliação

Métricas utilizadas: **WMAPE**.

O modelo final será avaliado com base na acurácia das previsões para as semanas de janeiro/2023.

## 👥 Equipe

- Rodrigo Lopes de Faria
- Rozana da Malta Martins