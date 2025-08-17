# analysis.py
"""
Módulo de análise para Pairs Trading
Funções estatísticas: cointegração, Z-Score, rolling beta, Kalman Filter, diagnósticos
+ Filtros avançados: ruptura estrutural, estabilidade do beta, volatilidade de mercado
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from typing import Tuple, Dict, Any
import warnings

# Suprimir warnings para não poluir o Streamlit
warnings.filterwarnings("ignore")

# Tentar importar yfinance (opcional, mas recomendado)
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("⚠️ yfinance não instalado. Filtros de dividendos e IBOV desativados.")


# Cache simples em memória
_dividend_cache = {}

def get_next_dividend_date(ticker: str) -> pd.Timestamp:
    """
    Obtém a próxima data de ex-dividendos usando yfinance.
    Usa cache para evitar múltiplas chamadas.
    Retorna None se não disponível.
    """
    if isinstance(ticker, np.ndarray):
        ticker = ticker.item() if ticker.size == 1 else str(ticker)
    ticker = str(ticker)

    if ticker in _dividend_cache:
        return _dividend_cache[ticker]

    if not YFINANCE_AVAILABLE:
        _dividend_cache[ticker] = None
        return None

    try:
        ticker_clean = ticker.replace(".SA", "")
        ticker_yf = f"{ticker_clean}.SA"
        ativo = yf.Ticker(ticker_yf)
        calendar = ativo.calendar

        if calendar is not None and 'Ex-Dividend Date' in calendar:
            ex_date = pd.to_datetime(calendar['Ex-Dividend Date'])
            ex_date = ex_date.tz_localize(None)
            _dividend_cache[ticker] = ex_date
            return ex_date
    except Exception as e:
        print(f"Erro ao buscar dividendo para {ticker}: {e}")

    _dividend_cache[ticker] = None
    return None


def check_fundamental_filters(s1: str, s2: str, data: pd.DataFrame) -> Dict[str, Any]:
    """
    Verifica filtros fundamentais para evitar operações em momentos ruins.
    """
    filters = {
        "dividendos": {"proximos_5d": False, "data": None, "status": "✅"},
        "eventos_corporativos": {"tipo": None, "data": None, "status": "✅"},
        "baixa_liquidez": {"status": "✅"},
        "suspensao": {"status": "✅"}
    }

    today = pd.Timestamp.today().normalize()
    for ticker in [s1, s2]:
        ex_date = get_next_dividend_date(ticker)
        if ex_date is not None and today <= ex_date <= today + pd.Timedelta(days=5):
            filters["dividendos"]["proximos_5d"] = True
            filters["dividendos"]["data"] = ex_date
            filters["dividendos"]["status"] = "⚠️"
            filters["eventos_corporativos"]["tipo"] = f"Dividendo {ticker}"
            filters["eventos_corporativos"]["status"] = "⚠️"

    try:
        vol_s1 = data[s1].pct_change().std() * np.sqrt(252)
        vol_s2 = data[s2].pct_change().std() * np.sqrt(252)
        if vol_s1 > 0.6 or vol_s2 > 0.6:
            if filters["eventos_corporativos"]["status"] == "✅":
                filters["eventos_corporativos"]["tipo"] = "Alta volatilidade"
                filters["eventos_corporativos"]["status"] = "⚠️"
    except:
        pass

    return filters


def calculate_rolling_beta(s1: pd.Series, s2: pd.Series, window: int = 60) -> pd.Series:
    """
    Calcula beta móvel entre s1 ~ s2 com janela rolante.
    """
    if window < 2:
        raise ValueError("A janela deve ser >= 2")
    common_idx = s1.index.intersection(s2.index)
    if len(common_idx) < window:
        return pd.Series([], dtype=float)
    s1 = s1.loc[common_idx]
    s2 = s2.loc[common_idx]
    betas = []
    dates = []
    for i in range(window, len(s1)):
        y = s1.iloc[i - window:i]
        X = s2.iloc[i - window:i]
        X = sm.add_constant(X)
        try:
            model = sm.OLS(y, X).fit()
            beta = model.params.iloc[1]
        except:
            beta = np.nan
        betas.append(beta)
        dates.append(s1.index[i])
    return pd.Series(betas, index=dates)


def kalman_filter_hedge_ratio(s1: pd.Series, s2: pd.Series, delta: float = 1e-3,
                              initial_state_cov: float = 1.0) -> pd.Series:
    """
    Estima o hedge ratio dinâmico usando Filtro de Kalman.
    
    Modelo: s1 = alpha + beta * s2 + epsilon
    Spread = s1 - beta * s2
    """
    try:
        common_idx = s1.index.intersection(s2.index)
        if len(common_idx) < 10:
            return pd.Series([], dtype=float)
        y = s1.loc[common_idx].values
        x = s2.loc[common_idx].values
        X = np.column_stack([np.ones(len(x)), x])
        R = 1.0
        delta = delta
        Qt = delta / (1 - delta) * np.eye(2)
        beta = np.zeros(2)
        P = np.eye(2) * initial_state_cov
        betas = []
        for t in range(len(y)):
            if t == 0:
                if len(y) >= 5:
                    beta_ols = np.linalg.pinv(X[:5].T @ X[:5]) @ X[:5].T @ y[:5]
                    beta = beta_ols
                    P = np.eye(2) * initial_state_cov
                else:
                    beta = np.array([0.0, 1.0])
            else:
                P = P + Qt
                pred = X[t] @ beta
                v = y[t] - pred
                F = X[t] @ P @ X[t] + R
                K = (P @ X[t]) / F
                beta = beta + K * v
                P = (np.eye(2) - np.outer(K, X[t])) @ P
            betas.append(beta[1])
        return pd.Series(betas, index=common_idx)
    except Exception as e:
        print(f"Kalman Filter erro: {e}")
        return pd.Series([], dtype=float)


def test_cointegration(s1: pd.Series, s2: pd.Series) -> Tuple[float, float, float]:
    """
    Teste de cointegração Engle-Granger com meia-vida.
    
    CONVENÇÃO: 
    - Modelo: s1 ~ s2
    - Spread = s1 - beta * s2
    - Z-Score negativo → s1 está barato → Long s1, Short s2
    """
    try:
        s1_clean = s1.dropna()
        s2_clean = s2.reindex(s1_clean.index).dropna()
        common_idx = s1_clean.index.intersection(s2_clean.index)
        if len(common_idx) < 30:
            return (np.nan, np.nan, np.inf)
        y = s1_clean.loc[common_idx]
        X = s2_clean.loc[common_idx]
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        residuals = model.resid
        adf_result = sm.tsa.stattools.adfuller(residuals, maxlag=1, autolag='AIC')
        pvalue = adf_result[1]
        spread_lag = residuals.shift(1)
        delta = residuals.diff().dropna()
        spread_lag = spread_lag.loc[delta.index]
        if len(spread_lag) < 2:
            hl = np.inf
        else:
            X_ar = sm.add_constant(spread_lag)
            model_ar = sm.OLS(delta, X_ar).fit()
            coef = model_ar.params.iloc[1]
            hl = -np.log(2) / coef if coef < 0 else np.inf
        beta_hedge = model.params.iloc[1]
        return (pvalue, beta_hedge, hl)
    except Exception as e:
        return (np.nan, np.nan, np.inf)


def calculate_zscore(spread: pd.Series, window: int = 20) -> pd.Series:
    """
    Calcula Z-Score móvel com média e desvio rolantes.
    """
    if len(spread) < window:
        return (spread - spread.mean()) / spread.std() if spread.std() > 0 else spread * 0.0
    mean = spread.rolling(window=window).mean()
    std = spread.rolling(window=window).std()
    std = std.replace(0, np.nan)
    zscore = (spread - mean) / std
    return zscore


def analyze_residuals_diagnostics(residuals: pd.Series) -> Dict[str, Any]:
    """
    Realiza diagnóstico completo dos resíduos do modelo de cointegração.
    """
    if len(residuals) < 10:
        return {"error": "Série muito curta"}
    diagnostics = {}
    try:
        from scipy.stats import shapiro
        _, p_shapiro = shapiro(residuals.dropna())
        diagnostics["normalidade"] = {
            "pvalor": p_shapiro,
            "normal": p_shapiro > 0.05,
            "status": "✅" if p_shapiro > 0.05 else "⚠️"
        }
    except:
        diagnostics["normalidade"] = {"normal": False, "status": "❌"}
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb_test = acorr_ljungbox(residuals.dropna(), lags=5, return_df=False)
        p_lb = np.min(lb_test[1])
        diagnostics["autocorrelacao"] = {
            "pvalor": p_lb,
            "sem_ac": p_lb > 0.05,
            "status": "✅" if p_lb > 0.05 else "⚠️"
        }
    except:
        diagnostics["autocorrelacao"] = {"sem_ac": False, "status": "❌"}
    try:
        from statsmodels.stats.diagnostic import het_breuschpagan
        X = sm.add_constant(np.arange(len(residuals)))
        bp_test = het_breuschpagan(residuals - residuals.mean(), X)
        p_bp = bp_test[1]
        diagnostics["heterocedasticidade"] = {
            "pvalor": p_bp,
            "homocedastico": p_bp > 0.05,
            "status": "✅" if p_bp > 0.05 else "⚠️"
        }
    except:
        diagnostics["heterocedasticidade"] = {"homocedastico": False, "status": "❌"}
    try:
        adf_stat, p_adf, _, _, _, _ = adfuller(residuals.dropna())
        diagnostics["estacionariedade"] = {
            "pvalor": p_adf,
            "estacionario": p_adf < 0.05,
            "status": "✅" if p_adf < 0.05 else "⚠️"
        }
    except:
        diagnostics["estacionariedade"] = {"estacionario": False, "status": "❌"}
    return diagnostics


# ========= NOVA FUNÇÃO 1: detect_structural_break =========
def detect_structural_break(residuals: pd.Series, threshold: float = 1.5) -> Dict[str, Any]:
    """
    Detecta rupturas estruturais nos resíduos usando CUSUM.
    """
    if len(residuals) < 20:
        return {"stable": True, "break_point": None, "status": "✅ Estável (poucos dados)"}
    
    # CUSUM simples
    std_resid = residuals.std()
    if std_resid == 0:
        return {"stable": False, "break_point": None, "status": "❌ Ruptura detectada"}
    
    cumsum = (residuals - residuals.mean()).cumsum() / std_resid
    upper_bound = threshold * np.sqrt(np.arange(len(cumsum)))
    lower_bound = -upper_bound
    
    # Verificar cruzamento
    if (cumsum > upper_bound).any() or (cumsum < lower_bound).any():
        idx = np.where((cumsum > upper_bound) | (cumsum < lower_bound))[0][0]
        date = residuals.index[idx]
        return {
            "stable": False,
            "break_point": date,
            "status": f"❌ Quebrado em {date.strftime('%d/%m')}"
        }
    
    return {"stable": True, "break_point": None, "status": "✅ Estável"}


# ========= NOVA FUNÇÃO 2: is_beta_stable =========
def is_beta_stable(beta_series: pd.Series, window: int = 30, max_std: float = 0.3) -> Dict[str, Any]:
    """
    Verifica se o beta estimado via Kalman está estável.
    """
    if len(beta_series) < window:
        return {"stable": True, "std": 0.0, "status": "✅ Estável (poucos dados)"}
    
    recent_beta = beta_series.iloc[-window:]
    beta_std = recent_beta.std()
    
    stable = beta_std <= max_std
    status = "✅ Estável" if stable else f"⚠️ Instável ({beta_std:.3f})"
    
    return {"stable": stable, "std": float(beta_std), "status": status}


# ========= NOVA FUNÇÃO 3: is_market_volatility_low =========
def is_market_volatility_low(threshold: float = 35.0) -> Dict[str, Any]:
    """
    Verifica se o mercado está em regime de baixa volatilidade com base no IBOV.
    """
    if not YFINANCE_AVAILABLE:
        return {"calm": True, "vol_20d": 0.0, "status": "🟡 Sem yfinance"}

    try:
        # Baixar dados do IBOV
        ibov_data = yf.download("^BVSP", period="3mo", interval="1d", progress=False)
        
        # ✅ Verificação segura: checar se está vazio
        if ibov_data.empty:
            return {"calm": True, "vol_20d": 0.0, "status": "🟡 Sem dados (IBOV)"}
        
        # ✅ Verificação segura: coluna "Close" existe e não está vazia
        if "Close" not in ibov_data or ibov_data["Close"].dropna().empty:
            return {"calm": True, "vol_20d": 0.0, "status": "🟡 Sem dados de fechamento"}

        close = ibov_data["Close"].dropna()
        
        if len(close) < 20:
            return {"calm": True, "vol_20d": 0.0, "status": "🟡 Poucos dados"}

        # ✅ Retornos diários
        returns = close.pct_change().dropna()
        
        if len(returns) < 20:
            return {"calm": True, "vol_20d": 0.0, "status": "🟡 Poucos retornos"}

        # ✅ Volatilidade anualizada dos últimos 20 dias
        vol_20d = returns.iloc[-20:].std() * np.sqrt(252) * 100  # em %

        # ✅ Conversão segura para float
        vol_20d = float(vol_20d)
        
        # ✅ Decisão clara
        calm = vol_20d <= threshold
        status = "🟢 Calmo" if calm else f"⚠️ Volátil ({vol_20d:.1f}%)"

        return {
            "calm": bool(calm),
            "vol_20d": round(vol_20d, 1),
            "status": status
        }

    except Exception as e:
        print(f"Erro ao calcular volatilidade do mercado: {e}")
        return {"calm": True, "vol_20d": 0.0, "status": "🟡 Erro ao checar"}
    
def diagnosticar_par(s1: pd.Series, s2: pd.Series, nome_s1: str, nome_s2: str):
    """
    Diagnóstico rápido para verificar se o z-score está coerente com os preços.
    """
    # Passo 1: calcular beta
    y = s1.dropna()
    x = s2.reindex(y.index).dropna()
    common_idx = y.index.intersection(x.index)
    y = y.loc[common_idx]
    x = x.loc[common_idx]

    if len(common_idx) < 20:
        print("⚠️ Poucos dados para diagnóstico")
        return

    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    beta = model.params.iloc[1]

    # Passo 2: calcular spread e z-score
    spread = y - beta * x
    zscore = (spread[-1] - spread.mean()) / spread.std()

    s1_atual = y.iloc[-1]
    s2_atual = x.iloc[-1]
    valor_justo_s1 = beta * s2_atual

    print(f"\n🔍 DIAGNÓSTICO DO PAR: {nome_s1} vs {nome_s2}")
    print(f"Beta: {beta:.3f}")
    print(f"{nome_s1} atual: R$ {s1_atual:.2f}")
    print(f"Valor justo (baseado em {nome_s2}): R$ {valor_justo_s1:.2f}")
    print(f"Z-Score atual: {zscore:.2f}")

    if zscore < -1.5:
        print(f"✅ Sinal: Long {nome_s1}, Short {nome_s2}")
    elif zscore > 1.5:
        print(f"✅ Sinal: Short {nome_s1}, Long {nome_s2}")
    else:
        print("🟡 Sem sinal claro")