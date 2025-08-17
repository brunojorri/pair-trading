# config.py
from datetime import datetime, timedelta


# Janelas de cálculo (em períodos - horários ou diários)
CORRELATION_WINDOW = 60          # Janela para cálculo de correlação
COINTEGRATION_WINDOW = 100       # Aumentado: cointegração requer mais estabilidade
ZSCORE_WINDOW = 40               # Janela para média e desvio padrão do z-score
ROLLING_BETA_WINDOW = 60         # Janela para beta rolante (usado como fallback)

# Critérios de seleção de pares
MIN_CORRELATION = 0.60           # Mínimo aceitável para fundos profissionais
MAX_PVALUE = 0.10                # P-valor máximo do teste ADF (cointegração)

# Filtros de qualidade do par
MIN_LIQUIDITY_BRL = 50_000_000   # Liquidez diária mínima por ativo (R$)
MAX_SPREAD_BID_ASK_RATIO = 0.01  # Spread relativo máximo (ex: 1%) - se disponível
MIN_TRADING_DAYS = 90            # Mínimo de dias de negociação para incluir ativo

# Parâmetros do Kalman Filter (estado da arte para hedge ratio dinâmico)
KALMAN_DELTA = 1e-5              # Força de adaptação (1e-4 = adaptação lenta e estável)
KALMAN_TRANSITION_COVARIANCE = 1e-5
KALMAN_OBSERVATION_COVARIANCE = 1.0

# Regras de entrada e saída
ZSCORE_ENTRY_THRESHOLD = 2.0     # |z| > 2.0 para entrada (simétrico)
ZSCORE_EXIT_THRESHOLD = 0.5      # |z| < 0.5 para saída
ZSCORE_STOP_LOSS = 3.0           # |z| > 3.0 → fechar posição (controle de risco)

# Custos operacionais (ajustáveis por ativo no futuro)
COMMISSION = 0.00025             # 0.025% por operação (ajustável)
SLIPPAGE = 0.0005                # 0.05% de deslizamento médio

# Controle de risco
MAX_POSITIONS_PER_PAIR = 1       # Apenas uma posição por par (long ou short)
MAX_DAILY_EXPOSURE_BRL = 1_000_000  # Exposição máxima diária (ex: R$1M)
DAILY_MAX_TRADES = 20            # Limite para evitar overtrading

# === Configurações de backtest: Últimos 60 dias em intervalo horário ===
def get_backtest_dates():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

BACKTEST_START_DATE, BACKTEST_END_DATE = get_backtest_dates()
BACKTEST_FREQUENCY = '1H'  # '1H' para horários, '1D' para diários

# Opcional: se quiser garantir que o end_date não vá além do que o Yahoo tem (ex: até ontem)
# BACKTEST_END_DATE = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

# Filtros de eventos corporativos
IGNORE_DIVIDEND_WINDOW = 5       # Evitar operar X dias antes/depois do ex-dividendo
FILTER_CORP_EVENTS = True        # Ativar filtro para splits, bonificações, etc.

# Diagnósticos estatísticos
RESIDUALS_NORMALITY_PVALUE = 0.05  # Mínimo para aceitar normalidade (Shapiro)
AUTOCORR_LJUNGBOX_PVALUE = 0.05    # Não deve haver autocorrelação
HETEROSKED_LM_PVALUE = 0.05        # Não deve haver heterocedasticidade

# Calendário de trading (B3)
TRADING_START_TIME = "10:00"
TRADING_END_TIME = "17:00"
TIMEZONE = "America/Sao_Paulo"
HOLIDAYS_FILE = "data/b3_holidays.csv"  # Caminho para feriados da B3

# Logging e saída
LOG_LEVEL = "INFO"
OUTPUT_DIR = "results"
PLOT_INTERACTIVE = True
SAVE_TRADES_TO_CSV = True