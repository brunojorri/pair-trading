#region # main_18_ok.py
"""
🔍 Sistema de Pairs Trading - B3
Versão 18_ok: Com temas personalizados, persistência, ícones e correções críticas
✅ Dark Mode / Light Mode com memória
✅ Tabela de Backtest corrigida
✅ Equity Curve interativa
✅ Página Watching funcional
"""

import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import numpy as np
import json
import os
import pickle
from datetime import datetime
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from typing import Tuple, Dict, Any
from config import KALMAN_DELTA
from analysis import kalman_filter_hedge_ratio
#endregion

#region # === CARREGAR CONFIGURAÇÃO ===
CONFIG_FILE = "config.json"

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"⚠️ Erro ao carregar config.json: {e}")
            return {
                "theme": "dark_blue",
                "tickers_file": "meus_tickers.json",
                "cache_ttl_hours": 1
            }

def save_config(config):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.warning(f"⚠️ Erro ao salvar config.json: {e}")

config = load_config()
#endregion

#region # === DEFINIÇÃO DE TEMAS (com cores hex corrigidas) ===
THEMES = {
    "light": {
        "primaryColor": "#007BFF",
        "backgroundColor": "#FFFFFF",    # ✅ Corrigido: de #FFFF para #FFFFFF
        "secondaryBackgroundColor": "#F0F2F6",
        "textColor": "#000000",          # ✅ Corrigido: de #0000 para #000000
        "icon": "☀️",
        "name": "Claro"
    },
    "bloomberg": {
        "primaryColor": "#001F3F",
        "backgroundColor": "#000E24",
        "secondaryBackgroundColor": "#000E24",
        "textColor": "#FFD7AA",
        "icon": "🌙",
        "name": "Azul Escuro"
    },
    "trader_green": {
        "primaryColor": "#00C853",
        "backgroundColor": "#001F1F",
        "secondaryBackgroundColor": "#003333",
        "textColor": "#CCFFCC",
        "icon": "🟢",
        "name": "Verde Trader"
    }
}

# Aplicar tema selecionado
current_theme = config.get("theme", "bloomberg")
if current_theme not in THEMES:
    current_theme = "bloomberg"
theme = THEMES[current_theme]

st._config.set_option("theme.primaryColor", theme["primaryColor"])
st._config.set_option("theme.backgroundColor", theme["backgroundColor"])
st._config.set_option("theme.secondaryBackgroundColor", theme["secondaryBackgroundColor"])
st._config.set_option("theme.textColor", theme["textColor"])
st._config.set_option("theme.font", "sans serif")
#endregion

#region # === IMPORTAR FUNÇÕES DE ANÁLISE ===
try:
    from analysis import (
        calculate_rolling_beta,
        test_cointegration,
        calculate_zscore,
        check_fundamental_filters,
        analyze_residuals_diagnostics,
        detect_structural_break,     # ✅ Nova função
        is_beta_stable,              # ✅ Nova função
        is_market_volatility_low     # ✅ Nova função
    )
except ImportError as e:
    st.error(f"❌ Não foi possível importar analysis.py: {e}")
    st.stop()

def para_exibicao(ticker: str) -> str:
    return ticker.replace(".SA", "") if isinstance(ticker, str) else ticker
#endregion

#region # === FUNÇÃO DE GRÁFICO DE Z-SCORE (VERSÃO FINAL — COM ESTILO + OPERAÇÕES) ===
def plot_zscore_chart(series, current_z, height=450, title="Z-Score", trade_log=None):
    if len(series) == 0:
        return go.Figure().update_layout(title="Sem dados", height=height)

    # Garantir datetime limpo
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)
    series.index = series.index.tz_localize(None)

    # Filtrar horário e dias úteis (opcional, mas bom para consistência)
    series = series.between_time("10:00", "17:00")
    series = series[series.index.dayofweek < 5]
    series = series.dropna()

    if len(series) == 0:
        return go.Figure().update_layout(title="Sem dados no horário de pregão", height=height)

    # Recortar para os últimos 60 pontos (ou menos)
    series = series.iloc[-60:]
    series_index = series.index.tolist()
    series_values = series.values
    x_numeric = list(range(len(series)))
    x_labels = [idx.strftime("%d/%m %H:%M") for idx in series_index]

    fig = go.Figure()

    # === Linha do Z-Score ===
    fig.add_trace(go.Scatter(
        x=x_numeric,
        y=series_values,
        mode='lines+markers',
        line=dict(color="#e7bd63", width=2.0),
        marker=dict(size=4, color="#f7c14e"),
        name="Z-Score"
    ))

    # === Linhas de limiar ===
    fig.add_hline(y=2, line_dash="dash", line_color="red", annotation_text="+2σ", annotation_font_color=theme["textColor"])
    fig.add_hline(y=-2, line_dash="dash", line_color="red", annotation_text="-2σ", annotation_font_color=theme["textColor"])
    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Média", annotation_font_color=theme["textColor"])
    fig.add_hline(y=3, line_dash="dot", line_color="darkred", annotation_text="+3σ (SL)", annotation_font_color=theme["textColor"], annotation_font_size=10)
    fig.add_hline(y=-3, line_dash="dot", line_color="darkred", annotation_text="-3σ (SL)", annotation_font_color=theme["textColor"], annotation_font_size=10)

    # === Destaque "Agora" ===
    fig.add_vrect(
        x0=x_numeric[-1] - 0.45,
        x1=x_numeric[-1] + 0.45,
        fillcolor="yellow",
        opacity=0.2,
        annotation_text="Agora",
        annotation_font_size=12,
        annotation_font_color="black"
    )

    # === MARCAÇÕES DE OPERAÇÕES (CORRIGIDO) ===
    if trade_log and len(trade_log) > 0:
        # Converter log para DataFrame com tempo limpo
        log_df = pd.DataFrame([t for t in trade_log if t.get('type') in ['entry', 'exit']])
        log_df['time'] = pd.to_datetime(log_df['time']).dt.tz_localize(None)

        # Filtrar apenas os sinais dentro do intervalo do gráfico
        start_time = series_index[0]
        end_time = series_index[-1]
        relevant_trades = log_df[(log_df['time'] >= start_time) & (log_df['time'] <= end_time)]

        # ✅ Mapear apenas os tempos que estão no recorte atual
        time_to_x = {t: i for i, t in enumerate(series_index)}

        # === Entradas Long (Z < -2) ===
        long_entries = relevant_trades[(relevant_trades['side'] == 'long_s1') & (relevant_trades['type'] == 'entry')]
        if not long_entries.empty:
            x_pts = [time_to_x[t] for t in long_entries['time'] if t in time_to_x]
            y_pts = [series.loc[t] for t in long_entries['time'] if t in time_to_x]
            fig.add_trace(go.Scatter(
                x=x_pts,
                y=y_pts,
                mode='markers',
                marker=dict(symbol='triangle-up', size=12, color='green', line=dict(width=1.5, color='white')),
                name='Entrada Long',
                hovertemplate='🟢 Long Entrada<br>Z = %{y:.2f}<extra></extra>'
            ))

        # === Entradas Short (Z > 2) ===
        short_entries = relevant_trades[(relevant_trades['side'] == 'long_s2') & (relevant_trades['type'] == 'entry')]
        if not short_entries.empty:
            x_pts = [time_to_x[t] for t in short_entries['time'] if t in time_to_x]
            y_pts = [series.loc[t] for t in short_entries['time'] if t in time_to_x]
            fig.add_trace(go.Scatter(
                x=x_pts,
                y=y_pts,
                mode='markers',
                marker=dict(symbol='triangle-down', size=12, color='purple', line=dict(width=1.5, color='white')),
                name='Entrada Short',
                hovertemplate='🟣 Short Entrada<br>Z = %{y:.2f}<extra></extra>'
            ))

        # === Saídas Long (TP ou SL) ===
        long_exits = relevant_trades[(relevant_trades['side'] == 'long_s1') & (relevant_trades['type'] == 'exit')]
        if not long_exits.empty:
            x_pts = [time_to_x[t] for t in long_exits['time'] if t in time_to_x]
            y_pts = [series.loc[t] for t in long_exits['time'] if t in time_to_x]
            colors = ['lime' if r != 'stop_loss' else 'red' for r in long_exits.get('exit_reason', ['take_profit'] * len(long_exits))]
            fig.add_trace(go.Scatter(
                x=x_pts,
                y=y_pts,
                mode='markers',
                marker=dict(symbol='x', size=10, color=colors, line=dict(width=1.5, color='white')),
                name='Saída Long',
                hovertemplate='🟢 Long Saída (%{text})<br>Z = %{y:.2f}<extra></extra>',
                text=['SL' if r == 'stop_loss' else 'TP' for r in long_exits.get('exit_reason', ['TP'] * len(long_exits))]
            ))

        # === Saídas Short (TP ou SL) ===
        short_exits = relevant_trades[(relevant_trades['side'] == 'long_s2') & (relevant_trades['type'] == 'exit')]
        if not short_exits.empty:
            x_pts = [time_to_x[t] for t in short_exits['time'] if t in time_to_x]
            y_pts = [series.loc[t] for t in short_exits['time'] if t in time_to_x]
            colors = ['orange' if r != 'stop_loss' else 'red' for r in short_exits.get('exit_reason', ['take_profit'] * len(short_exits))]
            fig.add_trace(go.Scatter(
                x=x_pts,
                y=y_pts,
                mode='markers',
                marker=dict(symbol='x', size=10, color=colors, line=dict(width=1.5, color='white')),
                name='Saída Short',
                hovertemplate='🟠 Short Saída (%{text})<br>Z = %{y:.2f}<extra></extra>',
                text=['SL' if r == 'stop_loss' else 'TP' for r in short_exits.get('exit_reason', ['TP'] * len(short_exits))]
            ))

    # === Layout final ===
    fig.update_layout(
        title=dict(text=f"{title} (atual: {current_z:.2f})", font=dict(color=theme["textColor"])),
        yaxis_range=[-3.5, 3.5],
        yaxis=dict(title="Z-Score", color=theme["textColor"]),
        xaxis=dict(
            title="",
            color=theme["textColor"],
            tickmode='array',
            tickvals=x_numeric[::5],
            ticktext=[x_labels[i] for i in range(0, len(x_labels), 5)],
            tickangle=0
        ),
        height=height,
        margin=dict(l=50, r=10, t=50, b=60),
        showlegend=True,
        legend=dict(font=dict(size=10), yanchor="top", y=0.98, xanchor="left", x=1.02),
        hovermode="x unified",
        plot_bgcolor=theme["backgroundColor"],
        paper_bgcolor=theme["backgroundColor"],
        font=dict(color=theme["textColor"])
    )

    return fig
#endregion

#region # === FUNÇÃO DE BACKTEST CORRIGIDA (SPREAD PADRONIZADO: S1 - β*S2) ===
def run_backtest(data_s1, data_s2, beta_inicial, entry_threshold=2.0, exit_threshold=0.5, stop_loss_threshold=3.5, slippage=0.0005, commission=0.00025):
    # Alinhar índices
    common_idx = data_s1.index.intersection(data_s2.index)
    data_s1 = data_s1.loc[common_idx]
    data_s2 = data_s2.loc[common_idx]

    # Beta dinâmico com Kalman
    beta_series = kalman_filter_hedge_ratio(data_s1, data_s2, delta=KALMAN_DELTA)
    beta_series = beta_series.reindex(data_s1.index).fillna(method='ffill').fillna(beta_inicial)
    beta_series = beta_series.shift(1).fillna(method='bfill')

    # Spread: S1 - β * S2
    spread = data_s1 - beta_series * data_s2
    zscore = calculate_zscore(spread, window=20).dropna()

    # Garantir que todos os dados estão alinhados no mesmo índice
    full_index = zscore.index
    data_s1 = data_s1.loc[full_index]
    data_s2 = data_s2.loc[full_index]
    beta_series = beta_series.loc[full_index]

    # Inicializar
    capital = 10000.0
    position = 0
    equity_curve = pd.Series(0.0, index=full_index)
    equity_curve.iloc[0] = capital
    trade_log = []

    # Iterar com base no índice
    for i in range(1, len(full_index)):
        current_time = full_index[i]
        prev_time = full_index[i-1]

        current_z = zscore.loc[current_time]
        price_s1 = data_s1.loc[current_time]
        price_s2 = data_s2.loc[current_time]
        current_beta = beta_series.loc[current_time]

        # === ENTRADA ===
        if position == 0:
            spread_vol = spread.rolling(window=20).std().loc[current_time]  # ✅ Volatilidade do spread
            risk_per_trade = capital * 0.02  # ✅ 2% de risco por trade

            if current_z <= -entry_threshold:  # Long S1 / Short S2
                q_s1 = risk_per_trade / (abs(current_z) * spread_vol)
                q_s2 = q_s1 * current_beta

                q_s1 = int(q_s1)
                q_s2 = int(q_s2)

                if q_s1 == 0 or q_s2 == 0:
                    equity_curve.loc[current_time] = capital
                    continue

                margem = q_s1 * price_s1
                position = 1

                # ✅ Append CORRETO com dicionário completo
                trade_log.append({
                    'type': 'entry',
                    'side': 'long_s1',
                    'zscore': current_z,
                    'price_s1': price_s1,
                    'price_s2': price_s2,
                    'q_s1': q_s1,
                    'q_s2': q_s2,
                    'margem': margem,
                    'time': current_time,
                    'beta': current_beta
                })

            elif current_z >= entry_threshold:  # Long S2 / Short S1
                q_s2 = risk_per_trade / (abs(current_z) * spread_vol)
                q_s1 = q_s2 * current_beta

                q_s1 = int(q_s1)
                q_s2 = int(q_s2)

                if q_s1 == 0 or q_s2 == 0:
                    equity_curve.loc[current_time] = capital
                    continue

                margem = q_s2 * price_s2
                position = -1

                # ✅ Append CORRETO com dicionário completo
                trade_log.append({
                    'type': 'entry',
                    'side': 'long_s2',
                    'zscore': current_z,
                    'price_s1': price_s1,
                    'price_s2': price_s2,
                    'q_s1': q_s1,
                    'q_s2': q_s2,
                    'margem': margem,
                    'time': current_time,
                    'beta': current_beta
                })

            elif current_z >= entry_threshold:  # Long S2 / Short S1
                q_s2 = risk_per_trade / (abs(current_z) * spread_vol)
                q_s1 = q_s2 * current_beta

                q_s1 = int(q_s1)
                q_s2 = int(q_s2)

                if q_s1 == 0 or q_s2 == 0:
                    equity_curve.loc[current_time] = capital
                    continue

                margem = q_s2 * price_s2
                position = -1

                trade_log.append({ ... })

        # === SAÍDA ===
        elif position != 0 and trade_log and trade_log[-1]['type'] == 'entry':
            entry = trade_log[-1]
            q_s1 = entry['q_s1']
            q_s2 = entry['q_s2']
            margem = entry['margem']

            take_profit = (position == 1 and current_z >= -exit_threshold) or \
                         (position == -1 and current_z <= exit_threshold)
            stop_loss = abs(current_z) >= stop_loss_threshold

            if take_profit or stop_loss:
                if position == 1:  # Long S1 / Short S2
                    pnl = (price_s1 - entry['price_s1']) * q_s1 - (price_s2 - entry['price_s2']) * q_s2
                else:  # Long S2 / Short S1
                    pnl = (price_s2 - entry['price_s2']) * q_s2 - (price_s1 - entry['price_s1']) * q_s1

                valor_total = (q_s1 * price_s1 + q_s2 * price_s2)
                custo = 2 * (slippage + commission) * valor_total
                pnl_liq = pnl - custo
                retorno_pct = (pnl_liq / margem) * 100 if margem > 0 else 0.0

                capital += pnl_liq
                equity_curve.loc[current_time] = capital

                trade_log.append({
                    'type': 'exit', 'zscore': current_z,
                    'pnl': pnl_liq, 'return_pct': retorno_pct,
                    'exit_reason': 'take_profit' if take_profit else 'stop_loss',
                    'price_s1': price_s1, 'price_s2': price_s2,
                    'time': current_time
                })
                position = 0
                continue

        # Manter capital se não houver mudança
        equity_curve.loc[current_time] = capital

    # === MÉTRICAS FINAIS ===
    equity_vals = equity_curve.values
    returns = np.diff(equity_vals) / equity_vals[:-1]
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 6.5) if len(returns) > 1 else 0
    max_dd = np.max((np.maximum.accumulate(equity_vals) - equity_vals) / np.maximum.accumulate(equity_vals)) if len(equity_vals) > 1 else 0
    win_rate = np.mean([t.get('pnl', 0) > 0 for t in trade_log if t['type'] == 'exit']) if trade_log else 0.0

    metrics = {
        'total_return': ((capital - 10000) / 10000) * 100,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'num_trades': sum(1 for t in trade_log if t['type'] == 'exit')
    }

    return metrics, trade_log, equity_curve, zscore, beta_series
#endregion

#region # === FUNÇÃO DE EQUITY CURVE ===
def plot_equity_curve(equity_curve, trade_log=None, title="Equity Curve"):
    """
    Plota a Equity Curve com eixo X baseado no tempo real.
    """
    fig = go.Figure()

    # ✅ USAR O ÍNDICE DO equity_curve (que é datetime)
    x_values = equity_curve.index  # ✅ Aqui está a correção
    y_values = equity_curve.values if isinstance(equity_curve, pd.Series) else equity_curve

    fig.add_trace(go.Scatter(
        x=x_values,  # ✅ Agora mostra datas/horas reais
        y=y_values,
        mode='lines',
        line=dict(color='cyan', width=2.8),
        name='Equity',
        hovertemplate='Equity: R$ %{y:.0f}<extra></extra>'
    ))

    # Opcional: adicionar área de drawdown (profissional)
    max_equity = np.maximum.accumulate(y_values)
    fig.add_trace(go.Scatter(
        x=x_values,
        y=max_equity,
        mode='lines',
        line=dict(color='rgba(0,255,255,0)', width=0),
        showlegend=False,
        hoverinfo='none'
    ))
    fig.add_trace(go.Scatter(
        x=x_values,
        y=y_values,
        mode='lines',
        line=dict(color='rgba(0,0,0,0)', width=0),
        fill='tonexty',
        fillcolor='rgba(255,0,0,0.1)',
        name='Drawdown',
        hoverinfo='none'
    ))

    # Layout
    fig.update_layout(
        title=dict(text=title, font=dict(color=theme["textColor"])),
        yaxis=dict(title="Valor (R$)", tickformat=",.0f", color=theme["textColor"]),
        xaxis=dict(
            title="",
            color=theme["textColor"],
            tickformat="%d/%m %H:%M",  # Formato claro: dia/mês hora:minuto
            tickangle=0
        ),
        height=400,
        margin=dict(l=50, r=10, t=50, b=60),
        plot_bgcolor=theme["backgroundColor"],
        paper_bgcolor=theme["backgroundColor"],
        font=dict(color=theme["textColor"]),
        showlegend=True,
        legend=dict(font=dict(size=10), yanchor="top", y=0.98, xanchor="left", x=1.02),
        hovermode="x unified"
    )

    return fig
#endregion

#region # === CARREGAMENTO DE DADOS (com cache) ===
@st.cache_data(ttl=1800)
def carregar_dados_1h(tickers):
    if not tickers:
        return pd.DataFrame()
    try:
        data = yf.download(tickers=tickers, period="60d", interval="1h", progress=False)["Close"]
        return data.dropna(axis=1, how="all")
    except Exception as e:
        st.error(f"❌ Erro ao baixar dados: {e}")
        return pd.DataFrame()
#endregion

#region # === CONFIGURAÇÃO DA PÁGINA ===
st.set_page_config(page_title="🔍 Pairs Trading B3", layout="wide")
#endregion

#region # === PRELOADER (mantido) ===
def carregar_tickers():
    ticker_file = config.get("tickers_file", "meus_tickers.json")
    if os.path.exists(ticker_file):
        try:
            with open(ticker_file, "r", encoding="utf-8") as f:
                return json.load(f).get("tickers", [])
        except Exception as e:
            st.error(f"❌ Erro ao ler {ticker_file}: {e}")
            return []
    else:
        st.warning(f"📭 Arquivo `{ticker_file}` não encontrado.")
        return []


tickers_salvos = carregar_tickers()

if "dados_pre_carregados" not in st.session_state:
    st.session_state.dados_pre_carregados = False

if tickers_salvos and not st.session_state.dados_pre_carregados:
    with st.spinner("🔍 Pré-carregando dados de mercado..."):
        dados_cache = carregar_dados_1h(tickers_salvos)
        if not dados_cache.empty:
            st.session_state.dados_pre_carregados = True
            st.rerun()
        else:
            st.warning("⚠️ Não foi possível carregar os dados.")
            st.stop()
#endregion

#region # === NAVEGAÇÃO ===
st.sidebar.markdown("<div style='padding: 20px 0;'>", unsafe_allow_html=True)
if st.sidebar.button("📊 Data", key="sidebar_data"):
    st.session_state.pagina = "Data"
if st.sidebar.button("⚙️ Health Check", key="sidebar_health"):
    st.session_state.pagina = "Health Check"
#if st.sidebar.button("🎯 Quality", key="sidebar_quality"):
#    st.session_state.pagina = "Quality"
#if st.sidebar.button("🧪 Backtest", key="sidebar_backtest"):
#    st.session_state.pagina = "Backtest"
if st.sidebar.button("👀 Watching", key="sidebar_watching"):
    st.session_state.pagina = "Watching"
st.sidebar.markdown("</div>", unsafe_allow_html=True)
st.sidebar.divider()
#endregion

#region # === PÁGINA POR PADRÃO ===
if "pagina" not in st.session_state:
    st.session_state.pagina = "Data"
pagina = st.session_state.pagina
#endregion

#region # === PÁGINA: DATA ===
if pagina == "Data":
    st.title("📊 Data — Cotações Horárias")
    if not tickers_salvos:
        st.info("⚠️ Nenhum ativo configurado. Verifique `meus_tickers.json`.")
        st.stop()

    with st.spinner("Carregando dados horários..."):
        data = carregar_dados_1h(tickers_salvos)

        if not data.empty:
            st.success(f"✅ Dados carregados: {data.shape[0]} períodos, {data.shape[1]} ativos")
            data_exib = data.iloc[::-1].copy()
            data_exib.columns = [para_exibicao(col) for col in data_exib.columns]

            st.write("### 🕐 Cotações Horárias (Últimos 60 dias)")
            st.info("🟢 Linha superior: cotação mais recente (última hora)")

            def destacar_primeira_linha(row):
                return ['background-color: #d4edda; color: #155724; font-weight: bold'] * len(row) if row.name == data_exib.index[0] else [''] * len(row)

            styled_data = data_exib.style.format("{:.2f}").apply(destacar_primeira_linha, axis=1)
            st.dataframe(styled_data, use_container_width=True, height=800)
        else:
            st.error("❌ Nenhum dado carregado.")
#endregion

#region # === PÁGINA: HEALTH CHECK ===
elif pagina == "Health Check":
    st.title("⚙️ Health Check — Análise Completa de Pares")
    if not tickers_salvos:
        st.stop()

    st.write("🔍 Análise de todos os pares possíveis")

    data = carregar_dados_1h(tickers_salvos)
    if data.empty:
        st.error("❌ Nenhum dado carregado.")
        st.stop()

    returns = data.pct_change().dropna()
    corr_matrix = returns.corr()
    pairs = []

    total_pairs = len(data.columns) * (len(data.columns) - 1) // 2
    progress_bar = st.progress(0)
    status_text = st.empty()

    idx = 0
    for i, s1 in enumerate(data.columns):
        for j, s2 in enumerate(data.columns):
            if i >= j:
                continue
            idx += 1
            status_text.text(f"Analisando par {idx}/{total_pairs}: {s1} vs {s2}")
            progress_bar.progress(idx / total_pairs)

            try:
                pvalue, beta, hl = test_cointegration(data[s1], data[s2])
                if pvalue > 0.3:
                    continue
                corr = corr_matrix.loc[s1, s2]
                spread = data[s1] - beta * data[s2]
                zscore_series = calculate_zscore(spread)
                current_z = zscore_series.iloc[-1] if len(zscore_series) > 0 else np.nan

                if np.isnan(pvalue) or np.isnan(beta) or np.isinf(hl) or hl < 0 or np.isnan(current_z):
                    continue

                pairs.append({
                    "Ativo 1": s1,
                    "Ativo 2": s2,
                    "Correlação": round(corr, 3),
                    "P-Valor": round(pvalue, 4),
                    "Beta": round(beta, 3),
                    "Meia-Vida (dias)": round(hl, 1),
                    "zscore_series": zscore_series,
                    "zscore_atual": current_z
                })
            except Exception as e:
                st.warning(f"Erro no par {s1} vs {s2}: {e}")
                continue

    progress_bar.empty()
    status_text.empty()

    if not pairs:
        st.warning("❌ Nenhum par analisado.")
        st.stop()

    # Paginação
    ITENS_POR_PAGINA = 9
    total_paginas = (len(pairs) // ITENS_POR_PAGINA) + 1
    if "pagina_health" not in st.session_state:
        st.session_state.pagina_health = 1

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("⬅️ Anterior", key="prev_health") and st.session_state.pagina_health > 1:
            st.session_state.pagina_health -= 1
    with col3:
        if st.button("Próxima ➡️", key="next_health") and st.session_state.pagina_health < total_paginas:
            st.session_state.pagina_health += 1
    st.write(f"**Página {st.session_state.pagina_health} de {total_paginas}**")

    inicio = (st.session_state.pagina_health - 1) * ITENS_POR_PAGINA
    fim = inicio + ITENS_POR_PAGINA
    pares_pagina = pairs[inicio:fim]

    for i in range(0, len(pares_pagina), 3):
        cols = st.columns(3)
        for j, pair in enumerate(pares_pagina[i:i+3]):
            if j < len(cols):
                with cols[j]:
                    a1 = para_exibicao(pair["Ativo 1"])
                    a2 = para_exibicao(pair["Ativo 2"])
                    z = pair["zscore_atual"]
                    emoji = "🔥 🔴" if z > 2 else "💧 🔵" if z < -2 else "⚠️ 🟡" if abs(z) >= 1.5 else "✅ 🟢"
                    st.markdown(f"### {emoji} {a1} vs {a2}")
                    fig = plot_zscore_chart(pair["zscore_series"], z, height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(f"β={pair['Beta']:.3f} | p={pair['P-Valor']:.4f}")
#endregion

#region # === PÁGINA: QUALITY ===
elif pagina == "Quality":
    st.title("🎯 Quality — Pares de Alta Qualidade Estatística")

    st.markdown("""
    🔍 Este painel mostra apenas os pares com **alta qualidade estatística e sinal claro de operação**.
    Os filtros estão configurados em modo rigoroso por padrão, mas podem ser ajustados abaixo.
    """)

    st.divider()
    st.subheader("🔧 Filtros de Seleção")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        MIN_CORRELATION = st.slider("Correlação mínima", 0.0, 0.9, 0.6, key="corr_slider")
    with col2:
        MAX_PVALUE = st.slider("P-Valor máximo", 0.01, 0.30, 0.10, key="pval_slider")
    with col3:
        MAX_HALF_LIFE = st.slider("Meia-Vida máxima (dias)", 30, 180, 60, key="hl_slider")
    with col4:
        MIN_ZSCORE_ABS = st.slider("Z-Score mínimo (abs)", 1.0, 3.0, 1.5, key="z_slider")

    # Segunda linha de filtros avançados
    col_beta, col_empty1, col_empty2, col_empty3 = st.columns(4)
    with col_beta:
        BETA_STABILITY_STD = st.slider("Máx desvio do Beta (30d)", 0.1, 0.8, 0.3, 0.1,
            help="Máximo desvio padrão do beta via Kalman para considerar estável")

    # === CARREGAR OU CALCULAR PARES ===
    CACHE_FILE = "pares_cache.pkl"
    CACHE_TTL_HOURS = config.get("cache_ttl_hours", 1)

    def should_use_cache():
        return os.path.exists(CACHE_FILE) and \
               (datetime.now() - datetime.fromtimestamp(os.path.getmtime(CACHE_FILE))).total_seconds() < CACHE_TTL_HOURS * 3600

    def load_cache():
        if should_use_cache():
            try:
                with open(CACHE_FILE, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                st.warning(f"⚠️ Falha ao carregar cache: {e}")
                return None
        return None

    def save_cache(data):
        try:
            with open(CACHE_FILE, "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            st.warning(f"⚠️ Falha ao salvar cache: {e}")

    todos_pares = load_cache()
    if not todos_pares:
        data_full = carregar_dados_1h(tickers_salvos)
        if data_full.empty:
            st.warning("⚠️ Nenhum dado carregado.")
            st.stop()

        with st.spinner("Analisando pares..."):
            returns = data_full.pct_change().dropna()
            corr_matrix = returns.corr()
            pairs = []
            total_pairs = len(data_full.columns) * (len(data_full.columns) - 1) // 2
            progress_bar = st.progress(0)
            status_text = st.empty()
            idx = 0

            for i, s1 in enumerate(data_full.columns):
                for j, s2 in enumerate(data_full.columns):
                    if i >= j:
                        continue
                    idx += 1
                    status_text.text(f"Analisando par {idx}/{total_pairs}: {s1} vs {s2}")
                    progress_bar.progress(idx / total_pairs)

                    try:
                        pvalue, beta, hl = test_cointegration(data_full[s1], data_full[s2])
                        if pvalue > 0.3:
                            continue
                        corr = corr_matrix.loc[s1, s2]
                        spread = data_full[s1] - beta * data_full[s2]
                        zscore_series = calculate_zscore(spread)
                        current_z = zscore_series.iloc[-1] if len(zscore_series) > 0 else np.nan

                        if (np.isnan(pvalue) or np.isnan(beta) or np.isinf(hl) or hl < 0 or np.isnan(current_z)):
                            continue

                        pairs.append({
                            "Ativo 1": s1,
                            "Ativo 2": s2,
                            "Correlação": round(corr, 3),
                            "P-Valor": round(pvalue, 4),
                            "Beta": round(beta, 3),
                            "Meia-Vida (dias)": round(hl, 1),
                            "zscore_series": zscore_series,
                            "zscore_atual": current_z
                        })
                    except Exception as e:
                        st.warning(f"Erro no par {s1} vs {s2}: {e}")
                        continue

            save_cache(pairs)
            status_text.text("✅ Análise concluída e salva em cache!")
            progress_bar.empty()
            status_text.empty()
            todos_pares = pairs
    else:
        st.info("✅ Análise carregada do cache.")
        data_full = carregar_dados_1h(tickers_salvos)

    # Verificação de mercado volátil (só alerta)
    market_state = is_market_volatility_low(threshold=35.0)
    if not market_state["calm"]:
        st.warning(f"⚠️ Mercado volátil (IBOV vol={market_state['vol_20d']}%). Novas entradas arriscadas.")

    # Aplicar filtros avançados
    pares_qualificados = []
    st.info("🔍 Aplicando filtros avançados: ruptura estrutural, estabilidade do beta...")

    for p in todos_pares:
        if not (
            p["Correlação"] >= MIN_CORRELATION
            and p["P-Valor"] <= MAX_PVALUE
            and p["Meia-Vida (dias)"] <= MAX_HALF_LIFE
            and abs(p["zscore_atual"]) >= MIN_ZSCORE_ABS
            and 0.5 <= p["Beta"] <= 2.0
            and not np.isnan(p["zscore_atual"])
        ):
            continue

        s1, s2 = p["Ativo 1"], p["Ativo 2"]
        common_idx = data_full[s1].index.intersection(data_full[s2].index)
        if len(common_idx) < 60:
            continue

        # Ruptura estrutural → usa apenas 'stable'
        model = sm.OLS(data_full[s1].loc[common_idx], sm.add_constant(data_full[s2].loc[common_idx])).fit()
        resid = model.resid
        sb_result = detect_structural_break(resid)
        if not sb_result["stable"]:
            continue

        # Estabilidade do beta
        beta_series = kalman_filter_hedge_ratio(data_full[s1], data_full[s2])
        beta_stable = is_beta_stable(beta_series, window=30, max_std=BETA_STABILITY_STD)
        if not beta_stable["stable"]:
            continue

        # ✅ Par passou em todos os filtros
        p["diagnostics"] = {
            "structural_break": sb_result,
            "beta_stability": beta_stable,
            "market_vol": market_state
        }
        pares_qualificados.append(p)

    pares_qualificados = sorted(pares_qualificados, key=lambda x: abs(x["zscore_atual"]), reverse=True)

    if pares_qualificados:
        st.success(f"✅ {len(pares_qualificados)} par(es) qualificados como TOP de operação")

        df_export = pd.DataFrame([
            {
                "Ativo 1": para_exibicao(p["Ativo 1"]),
                "Ativo 2": para_exibicao(p["Ativo 2"]),
                "Z-Score": round(p["zscore_atual"], 2),
                "Beta": round(p["Beta"], 3),
                "P-Valor": round(p["P-Valor"], 4),
                "Meia-Vida": round(p["Meia-Vida (dias)"], 1),
                "Correlação": round(p["Correlação"], 3),
                "Sinal": f"Long {p['Ativo 1']} / Short {p['Ativo 2']}" if p["zscore_atual"] < 0 else f"Long {p['Ativo 2']} / Short {p['Ativo 1']}"
            }
            for p in pares_qualificados
        ])
        st.download_button(
            "📥 Exportar TOP Pares (CSV)",
            df_export.to_csv(index=False).encode("utf-8"),
            "top_pairs_quality.csv",
            "text/csv"
        )

        n_cols = 3
        cols = st.columns(n_cols)
        idx = 0
        for pair in pares_qualificados[:12]:
            a1 = para_exibicao(pair["Ativo 1"])
            a2 = para_exibicao(pair["Ativo 2"])
            z = pair["zscore_atual"]
            emoji = "🔥 🔴" if z > 2 else "💧 🔵" if z < -2 else "⚠️ 🟡"
            beta = pair["Beta"]
            hl = pair["Meia-Vida (dias)"]
            p_val = pair["P-Valor"]
            corr = pair["Correlação"]
            long = a1 if z < 0 else a2
            short = a2 if z < 0 else a1

            with cols[idx % n_cols]:
                st.markdown(f"### {emoji} {a1} vs {a2}")
                st.markdown(f"<p style='color: #aaa; font-size: 0.9em; margin-top: -10px;'>Z: {z:.2f}</p>", unsafe_allow_html=True)

                fig = plot_zscore_chart(series=pair["zscore_series"], current_z=z, height=450, title="")
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("📊 Detalhes da Operação", expanded=False):
                    st.markdown("### 🔍 Estatísticas do Par")
                    col_x1, col_x2, col_x3 = st.columns(3)
                    with col_x1:
                        st.metric("Correlação", f"{corr:.3f}")
                    with col_x2:
                        st.metric("P-Valor", f"{p_val:.4f}")
                    with col_x3:
                        st.metric("Meia-Vida", f"{hl:.1f}d")

                    st.markdown("### 📌 Sinal Atual")
                    col_y1, col_y2 = st.columns([2, 1])
                    with col_y1:
                        st.markdown(
                            f"<h4 style='color: #e67e22;'>✅ Long {long}</h4>"
                            f"<span style='color: #aaa; font-size: 0.9em;'>(Short {short})</span>",
                            unsafe_allow_html=True
                        )
                    with col_y2:
                        st.markdown(
                            f"<h4 style='color: #3498db; text-align: center;'>β<br><span style='font-size: 1.2em;'>{beta:.3f}</span></h4>",
                            unsafe_allow_html=True
                        )

                with st.expander("🔍 Diagnóstico de Resíduos", expanded=False):
                    model = sm.OLS(data_full[pair["Ativo 1"]], sm.add_constant(data_full[pair["Ativo 2"]])).fit()
                    residuals = model.resid
                    diag = analyze_residuals_diagnostics(residuals)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Normalidade**: {diag['normalidade']['status']} {'Sim' if diag['normalidade']['normal'] else 'Não'}")
                    with col2:
                        st.markdown(f"**Sem Autocorr.**: {diag['autocorrelacao']['status']} {'Sim' if diag['autocorrelacao']['sem_ac'] else 'Não'}")
                    with col1:
                        st.markdown(f"**Homocedástico**: {diag['heterocedasticidade']['status']} {'Sim' if diag['heterocedasticidade']['homocedastico'] else 'Não'}")
                    with col2:
                        st.markdown(f"**Estacionário**: {diag['estacionariedade']['status']} {'Sim' if diag['estacionariedade']['estacionario'] else 'Não'}")

                # ✅ Mostrar diagnósticos avançados
                st.markdown("### 🔐 Filtros Avançados")
                col_d1, col_d2, col_d3 = st.columns(3)
                with col_d1:
                    st.markdown(f"**Ruptura Estrutural**<br>{pair['diagnostics']['structural_break']['status']}", unsafe_allow_html=True)
                with col_d2:
                    st.markdown(f"**Beta Estável**<br>{pair['diagnostics']['beta_stability']['status']}", unsafe_allow_html=True)
                with col_d3:
                    st.markdown(f"**Vol. Mercado**<br>{pair['diagnostics']['market_vol']['status']}", unsafe_allow_html=True)

            idx += 1
    else:
        st.warning("❌ Nenhum par passou nos critérios atuais.")
        st.info("💡 Dica: Ajuste os sliders acima para relaxar os filtros temporariamente.")
#endregion

#region # === PÁGINA: BACKTEST ===
elif pagina == "Backtest":
    st.title("🧪 Backtest — Simulação Histórica do Par")
    st.write("Simulação de operações passadas com base no Z-Score e Kalman Filter")

    data_full = carregar_dados_1h(tickers_salvos)
    if data_full.empty:
        st.warning("⚠️ Nenhum dado carregado.")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        ativo1 = st.selectbox("Ativo 1 (base)", data_full.columns, key="bt_ativo1")
    with col2:
        ativo2 = st.selectbox("Ativo 2 (cotado)", data_full.columns, key="bt_ativo2")

    if ativo1 == ativo2:
        st.warning("⚠️ Selecione ativos diferentes.")
        st.stop()

    if ativo1 not in data_full.columns or ativo2 not in data_full.columns:
        st.error("❌ Ativos inválidos.")
        st.stop()

    data_s1 = data_full[ativo1]
    data_s2 = data_full[ativo2]
    common_idx = data_s1.index.intersection(data_s2.index)
    data_s1 = data_s1.loc[common_idx]
    data_s2 = data_s2.loc[common_idx]

    pvalue, beta_inicial, hl = test_cointegration(data_s1, data_s2)
    st.info(f"📊 Cointegração: p={pvalue:.4f} | β={beta_inicial:.3f} | Meia-Vida={hl:.1f} dias")

    # Configurações do backtest
    entry_threshold = st.slider("Limiar de Entrada (σ)", 1.0, 3.0, 2.0, step=0.1, key="bt_entry")
    exit_threshold = st.slider("Limiar de Saída (σ)", 0.1, 2.0, 0.5, step=0.1, key="bt_exit")
    stop_loss_threshold = st.slider("Stop Loss (σ)", 2.0, 6.0, 3.0, step=0.1, key="bt_stop")
    slippage = st.slider("Slippage (bps)", 0.0, 10.0, 5.0, step=0.5, key="bt_slippage") / 10000
    commission = st.slider("Comissão (bps)", 0.0, 10.0, 2.5, step=0.5, key="bt_commission") / 10000

    if st.button("🚀 Executar Backtest", key="run_backtest"):
        with st.spinner("Rodando backtest..."):
            # Executa o backtest com as correções
            metrics, trade_log, equity_curve, zscore_series, beta_series = run_backtest(
                data_s1, data_s2, beta_inicial,
                entry_threshold=entry_threshold,
                exit_threshold=exit_threshold,
                stop_loss_threshold=stop_loss_threshold,
                slippage=slippage,
                commission=commission
            )
            # === RESULTADOS ===
            st.subheader("📊 Resultados do Backtest")
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Retorno Total", f"{metrics['total_return']:.1f}%")
            with col2: st.metric("Índice de Sharpe", f"{metrics['sharpe_ratio']:.2f}")
            with col3: st.metric("Máx. Drawdown", f"{metrics['max_drawdown']:.1%}")
            with col4: st.metric("Win Rate", f"{metrics['win_rate']:.1%}")

            # Equity Curve (opcional)
            st.plotly_chart(plot_equity_curve(equity_curve), use_container_width=True)

            # === GRÁFICO DE Z-SCORE COM OPERAÇÕES ===
            st.write("### 📈 Z-Score com Sinais de Operações")
            fig = plot_zscore_chart(
                series=zscore_series,
                current_z=zscore_series.iloc[-1],
                title="Z-Score com Operações",
                trade_log=trade_log  # ✅ Passa o log para marcar sinais
            )
            st.plotly_chart(fig, use_container_width=True)

            # Log de operações com motivo de saída
            if trade_log:
                st.write("### 📝 Log de Operações")
                trades_df = pd.DataFrame([
                    {
                        "Tipo": t['type'].title(),
                        "Lado": t.get('side', '').title(),
                        "Z-Score": f"{t['zscore']:.2f}",
                        "Retorno (%)": f"{t.get('return_pct', 0):+.2f}%" if t['type'] == 'exit' else "-",
                        "Motivo": t.get('exit_reason', '-').upper() if t['type'] == 'exit' else "-",
                        "Margem (R$)": f"R$ {t.get('margem_entrada', 0):.2f}" if t['type'] == 'entry' else "-"
                    }
                    for t in trade_log
                ])
                st.dataframe(trades_df, use_container_width=True, height=400)

                # Estatísticas por tipo de saída
                exit_types = [t.get('exit_reason', 'take_profit') for t in trade_log if t['type'] == 'exit']
                if 'stop_loss' in exit_types:
                    win_rate_sl = np.mean([t['pnl'] > 0 for t in trade_log if t.get('exit_reason') == 'stop_loss'])
                    st.caption(f"📉 **Win Rate em Stop Loss**: {win_rate_sl:.1%} ({exit_types.count('stop_loss')} saídas)")

                # Após carregar os dados
                data_full = carregar_dados_1h(tickers_salvos)
                beta_series = kalman_filter_hedge_ratio(data_full['BBAS3.SA'], data_full['BBSE3.SA'], delta=KALMAN_DELTA)
                print(f"Beta médio (Kalman): {beta_series.mean():.3f}")
                print(f"Desvio do beta: {beta_series.std():.3f}")
#endregion

#region # === PÁGINA: WATCHING (COM TABELA DINÂMICA E DIAGNÓSTICO AVANÇADO) ===
elif pagina == "Watching":
    st.title("👀 Watching — Meus Pares Favoritos")

    st.markdown("""
    🔍 Acompanhe seus pares favoritos com gráficos, sinais e diagnósticos avançados.
    Todos os pares são exibidos sequencialmente para melhor visualização.
    """)

    # === AVISO DE VOLATILIDADE DE MERCADO (NOVO) ===
    market_state = is_market_volatility_low(threshold=35.0)
    if not market_state["calm"]:
        st.info("🟢 Mercado calmo: condições favoráveis para pares estatísticos")
    else:
        st.warning(f"⚠️ Mercado volátil (IBOV vol={market_state['vol_20d']}%). Pares ainda são exibidos, mas novas entradas arriscadas.")


    FAVORITOS_FILE = "favoritos.json"

    def carregar_favoritos():
        if os.path.exists(FAVORITOS_FILE):
            try:
                with open(FAVORITOS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("pares_salvos", [])
            except Exception as e:
                st.warning(f"⚠️ Erro ao carregar favoritos: {e}")
                return []
        return []

    def salvar_favorito(par):
        favoritos = carregar_favoritos()
        favoritos.append(par)
        with open(FAVORITOS_FILE, "w", encoding="utf-8") as f:
            json.dump({"pares_salvos": favoritos}, f, indent=2, ensure_ascii=False)

    def remover_favorito(idx):
        favoritos = carregar_favoritos()
        if 0 <= idx < len(favoritos):
            favoritos.pop(idx)
            with open(FAVORITOS_FILE, "w", encoding="utf-8") as f:
                json.dump({"pares_salvos": favoritos}, f, indent=2, ensure_ascii=False)

    favoritos = carregar_favoritos()

    # === MONTAR NOVO PAR ===
    st.divider()
    st.subheader("➕ Criar Novo Par")

    col1, col2, col3 = st.columns([3, 3, 4])
    with col1:
        ativo1 = st.selectbox("Ativo 1", tickers_salvos, index=0, key="watch_ativo1")
    with col2:
        ativo2 = st.selectbox("Ativo 2", tickers_salvos, index=1, key="watch_ativo2")
    with col3:
        nome_par = st.text_input("Nome (opcional)", placeholder="Ex: Vale vs Petrobras", key="watch_nome")

    if st.button("💾 Salvar Par", key="watch_salvar"):
        if ativo1 == ativo2:
            st.warning("⚠️ Os ativos devem ser diferentes!")
        else:
            par = {
                "ativo1": ativo1,
                "ativo2": ativo2,
                "nome": nome_par.strip() or f"{para_exibicao(ativo1)} vs {para_exibicao(ativo2)}"
            }
            salvar_favorito(par)
            st.success(f"✅ Par '{par['nome']}' salvo!")
            st.rerun()

    # === TABELA DE ALOCAÇÃO COM LOTES E P&L (SEM R/R) ===
    st.divider()
    st.subheader("🎯 Pares Favoritos — Retorno Esperado com Base na Margem")

    lote_minimo = 100
    capital_base = st.number_input(
        "Capital base por par (R$)",
        min_value=1000,
        max_value=50000,
        value=10000,
        step=100,
        help="Usado para calcular número de lotes com base no beta dinâmico"
    )

    if not favoritos:
        st.info("📭 Nenhum par salvo ainda.")
    else:
        data_full = carregar_dados_1h(tickers_salvos)
        if data_full.empty:
            st.warning("⚠️ Dados não carregados. Vá para 'Data' primeiro.")
        else:
            tabela = []
            for par in favoritos:
                a1, a2 = par["ativo1"], par["ativo2"]
                nome = par["nome"]

                if a1 not in data_full.columns or a2 not in data_full.columns:
                    continue

                try:
                    # Beta dinâmico com Kalman Filter
                    beta_series = kalman_filter_hedge_ratio(data_full[a1], data_full[a2], delta=KALMAN_DELTA)
                    current_beta = beta_series.iloc[-1] if len(beta_series) > 0 else 1.0

                    # Teste de cointegração
                    pvalue, beta, hl = test_cointegration(data_full[a1], data_full[a2])

                    # Spread e Z-Score
                    spread = data_full[a1] - current_beta * data_full[a2]
                    zscore_series = calculate_zscore(spread)
                    current_z = zscore_series.iloc[-1] if len(zscore_series) > 0 else np.nan

                    # Preços atuais
                    p1 = data_full[a1].iloc[-1]
                    p2 = data_full[a2].iloc[-1]

                    # Calcular lotes com base no capital e beta
                    valor_lote_a2 = lote_minimo * p2
                    lotes_a2 = max(1, int(capital_base / valor_lote_a2))
                    lotes_a1 = max(1, int(lotes_a2 * abs(current_beta)))

                    # Definir sinais com base no Z-Score
                    q1 = -lotes_a1 * lote_minimo  # vende A1
                    q2 = lotes_a2 * lote_minimo   # compra A2
                    if current_z < 0:
                        q1, q2 = -q1, -q2  # inverte: compra A1, vende A2

                    # Valores em reais com sinal
                    valor_a1 = q1 * p1
                    valor_a2 = q2 * p2

                    # Calcular margem (lado longo - lado curto)
                    valor_a1_abs = abs(valor_a1)
                    valor_a2_abs = abs(valor_a2)
                    valor_longo = valor_a1_abs if q1 > 0 else valor_a2_abs
                    valor_curto = valor_a2_abs if q1 > 0 else valor_a1_abs
                    margem = valor_longo - valor_curto
                    margem_str = f"{'-' if margem < 0 else ''}R$ {abs(margem):,.2f}"

                    # P&L estimado: reversão de |Z| = 2 → 0
                    vol_spread = spread.std()
                    pnl_estimado = 0.0
                    if not np.isnan(vol_spread) and vol_spread > 0:
                        pnl_estimado = 2.0 * vol_spread * abs(q1)  # total em R$
                        pnl_estimado = round(pnl_estimado, 2)

                    pnl_color = "🟢" if pnl_estimado > 0 else "🟡"
                    pnl_str = f"{pnl_color} R$ {pnl_estimado:,.2f}"

                    # Destaque: |Z| >= 2
                    em_oportunidade = abs(current_z) >= 2.0

                    tabela.append({
                        "Par": nome,
                        "Z-Score": round(current_z, 2),
                        "Beta (Kalman)": round(current_beta, 3),
                        "Valor A1 (R$)": valor_a1,
                        "Valor A2 (R$)": valor_a2,
                        "Margem (R$)": margem_str,
                        "P&L Estimado": pnl_str,
                        "p-valor": f"{pvalue:.4f}",
                        "em_oportunidade": em_oportunidade
                    })

                except Exception as e:
                    continue

            if not tabela:
                st.warning("⚠️ Nenhum par pôde ser avaliado.")
            else:
                df_tabela = pd.DataFrame(tabela)
                if df_tabela.empty:
                    st.warning("⚠️ Nenhum dado válido.")
                else:
                    # Ordenar por |Z| decrescente
                    df_tabela["abs_z"] = df_tabela["Z-Score"].abs()
                    df_tabela = df_tabela.sort_values("abs_z", ascending=False).drop(columns=["abs_z"])

                    # Destaque visual: |Z| >= 2 → fundo verde escuro
                    def destacar_linha(row):
                        if row["em_oportunidade"]:
                            return ["background-color: #003300; color: #ccffcc; font-weight: bold"] * len(row)
                        return [""] * len(row)

                    # Formatação
                    styled = df_tabela.style.format({
                        "Z-Score": "{:.2f}",
                        "Beta (Kalman)": "{:.3f}",
                        "Valor A1 (R$)": lambda x: f"R$ {x:,.2f}",
                        "Valor A2 (R$)": lambda x: f"R$ {x:,.2f}",
                    }).apply(destacar_linha, axis=1)

                    st.dataframe(styled, use_container_width=True, height=500)

                st.caption("""
                🟩 Linha verde: |Z| ≥ 2 (oportunidade operacional) | 💡 P&L estimado baseado na reversão de |Z|=2 → 0
                """)

    # === LISTA DE PARES SALVOS (UM POR LINHA, EM CARDS) ===
    st.divider()
    st.subheader("📊 Pares Salvos")

    if not favoritos:
        st.info("📭 Nenhum par salvo ainda. Monte um acima!")
    else:
        data_full = carregar_dados_1h(tickers_salvos)
        if data_full.empty:
            st.warning("⚠️ Dados não carregados. Vá para 'Data' primeiro.")
        else:
            for idx, par in enumerate(favoritos):
                a1, a2 = par["ativo1"], par["ativo2"]
                nome = par["nome"]

                with st.container():
                    # Header do card
                    col_title, col_remove = st.columns([10, 1])
                    with col_title:
                        st.markdown(f"### 📈 {nome}")
                        st.markdown(f"<p style='color: #aaa; font-size: 0.9em; margin-top: -10px;'>{para_exibicao(a1)} vs {para_exibicao(a2)}</p>", unsafe_allow_html=True)
                    with col_remove:
                        if st.button("🗑️", key=f"del_fav_{idx}", help="Remover par"):
                            remover_favorito(idx)
                            st.rerun()

                    # Verificação de dados
                    if a1 not in data_full.columns or a2 not in data_full.columns:
                        st.error(f"❌ Dados faltando para {a1} ou {a2}")
                        st.divider()
                        continue

                    # Cálculos estatísticos
                    try:
                        pvalue, beta, hl = test_cointegration(data_full[a1], data_full[a2])
                        spread = data_full[a2] - beta * data_full[a1]
                        zscore_series = calculate_zscore(spread)
                        current_z = zscore_series.iloc[-1] if len(zscore_series) > 0 else np.nan
                    except Exception as e:
                        st.error(f"❌ Erro ao calcular estatísticas: {e}")
                        st.divider()
                        continue

                    # Z-Score atual + Beta + p-valor
                    emoji = "🔥 🔴" if current_z > 2 else "💧 🔵" if current_z < -2 else "⚠️ 🟡" if abs(current_z) >= 1.5 else "✅ 🟢"
                    st.markdown(f"**Z-Score Atual:** `{current_z:.2f}` {emoji} | **Beta:** `{beta:.3f}` | **p-valor:** `{pvalue:.4f}`")

                    # Gráfico de Z-Score
                    fig = plot_zscore_chart(zscore_series, current_z, height=400, title="")
                    st.plotly_chart(fig, use_container_width=True)

                    # Métricas rápidas
                    col_i1, col_i2, col_i3 = st.columns(3)
                    with col_i1:
                        corr = data_full[a1].pct_change().corr(data_full[a2].pct_change())
                        st.metric("Correlação", f"{corr:.3f}")
                    with col_i2:
                        st.metric("Meia-Vida", f"{hl:.1f} dias")
                    with col_i3:
                        st.metric("Beta", f"{beta:.3f}")

                    # Sinal de operação
                    long = para_exibicao(a1) if current_z < 0 else para_exibicao(a2)
                    short = para_exibicao(a2) if current_z < 0 else para_exibicao(a1)
                    st.markdown("### 📌 Sinal Atual")
                    col_y1, col_y2 = st.columns([2, 1])
                    with col_y1:
                        st.markdown(
                            f"<h4 style='color: #e67e22;'>✅ Long {long}</h4>"
                            f"<span style='color: #aaa; font-size: 0.9em;'>(Short {short})</span>",
                            unsafe_allow_html=True
                        )
                    with col_y2:
                        st.markdown(
                            f"<h4 style='color: #3498db; text-align: center;'>β<br><span style='font-size: 1.2em;'>{beta:.3f}</span></h4>",
                            unsafe_allow_html=True
                        )

                    # === DIAGNÓSTICO DE RESÍDUOS AVANÇADO (COM BETA ESTÁVEL E RUPTURA) ===
                    st.markdown("### 🔍 Diagnóstico de Resíduos Avançado")

                    try:
                        # Diagnóstico padrão
                        model = sm.OLS(data_full[a1], sm.add_constant(data_full[a2])).fit()
                        residuals = model.resid  # ✅ Variável correta
                        diag = analyze_residuals_diagnostics(residuals)

                        # Estabilidade do beta
                        beta_series = kalman_filter_hedge_ratio(data_full[a1], data_full[a2])
                        beta_stable = is_beta_stable(beta_series, window=30, max_std=0.3)

                        # Ruptura estrutural → ✅ CORRIGIDO: usa 'residuals', não 'resid'
                        sb_result = detect_structural_break(residuals)

                        # Exibir em 4 colunas + 1 extra para novos
                        col_d1, col_d2, col_d3, col_d4, col_d5, col_d6, col_d7 = st.columns(7)
                        with col_d1:
                            status = "✅ Sim" if diag['normalidade']['normal'] else "❌ Não"
                            color = "green" if diag['normalidade']['normal'] else "red"
                            st.markdown(f"**Normalidade**<br><span style='color:{color};'>{status}</span>", unsafe_allow_html=True)
                        with col_d2:
                            status = "✅ Sim" if diag['autocorrelacao']['sem_ac'] else "❌ Não"
                            color = "green" if diag['autocorrelacao']['sem_ac'] else "red"
                            st.markdown(f"**Sem Autocorr.**<br><span style='color:{color};'>{status}</span>", unsafe_allow_html=True)
                        with col_d3:
                            status = "✅ Sim" if diag['heterocedasticidade']['homocedastico'] else "❌ Não"
                            color = "green" if diag['heterocedasticidade']['homocedastico'] else "red"
                            st.markdown(f"**Homocedástico**<br><span style='color:{color};'>{status}</span>", unsafe_allow_html=True)
                        with col_d4:
                            status = "✅ Sim" if diag['estacionariedade']['estacionario'] else "❌ Não"
                            color = "green" if diag['estacionariedade']['estacionario'] else "red"
                            st.markdown(f"**Estacionário**<br><span style='color:{color};'>{status}</span>", unsafe_allow_html=True)
                        with col_d5:
                            status = "✅ Estável" if beta_stable["stable"] else "⚠️ Instável"
                            color = "green" if beta_stable["stable"] else "orange"
                            st.markdown(f"**Estab. Beta**<br><span style='color:{color};'>{status}</span>", unsafe_allow_html=True)
                        with col_d6:
                            status = "✅ Estável" if sb_result["stable"] else "❌ Quebrado"
                            color = "green" if sb_result["stable"] else "red"
                            st.markdown(f"**Ruptura**<br><span style='color:{color};'>{status}</span>", unsafe_allow_html=True)
                        with col_d7:
                            status = "🟢 Calmo" if market_state["calm"] else "⚠️ Volátil"
                            color = "green" if market_state["calm"] else "orange"
                            st.markdown(f"**Vol. Mercado**<br><span style='color:{color};'>{status}</span>", unsafe_allow_html=True)

                    except Exception as e:
                        st.warning(f"⚠️ Erro no diagnóstico avançado: {e}")

                    # === FILTROS FUNDAMENTAIS ===
                    st.markdown("### 📌 Filtros Fundamentais")
                    try:
                        filtros = check_fundamental_filters(a1, a2, data_full)
                        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
                        with col_f1:
                            status = filtros['dividendos']['status']
                            color = "green" if "✅" in status else "red"
                            st.markdown(f"**Dividendos (5d)**<br><span style='color:{color};'>{status}</span>", unsafe_allow_html=True)
                            if filtros['dividendos']['proximos_5d']:
                                data_div = filtros['dividendos']['data']
                                st.caption(f"📅 {data_div.strftime('%d/%m')}")

                        with col_f2:
                            status = filtros['eventos_corporativos']['status']
                            color = "green" if "✅" in status else "red"
                            st.markdown(f"**Eventos Corp.**<br><span style='color:{color};'>{status}</span>", unsafe_allow_html=True)
                            tipo = filtros['eventos_corporativos']['tipo']
                            if tipo:
                                st.caption(f"📁 {tipo}")

                        with col_f3:
                            status = filtros['baixa_liquidez']['status']
                            color = "green" if "✅" in status else "red"
                            st.markdown(f"**Liquidez**<br><span style='color:{color};'>{status}</span>", unsafe_allow_html=True)

                        with col_f4:
                            status = filtros['suspensao']['status']
                            color = "green" if "✅" in status else "red"
                            st.markdown(f"**Suspensão**<br><span style='color:{color};'>{status}</span>", unsafe_allow_html=True)

                    except Exception as e:
                        st.warning(f"⚠️ Erro nos filtros fundamentais: {e}")

                    st.divider()  # Separador entre cards
#endregion