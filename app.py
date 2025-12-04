import streamlit as st
import warnings
import tempfile
import os
import datetime

# --------------------------
# 1. Page Config & Custom Styling
# --------------------------
st.set_page_config(
    page_title="AI-Driven Stock Analysis - Advaith Parimisetti", 
    layout="wide",
    page_icon="📈"
)
warnings.filterwarnings("ignore")

# --- CUSTOM CSS INJECTION ---
st.markdown("""
<style>
    /* Main Background & Text */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Inputs */
    .stTextInput>div>div>input {
        background-color: #262730;
        color: white;
        border: 1px solid #4B5563;
        border-radius: 5px;
    }
    .stSelectbox>div>div>div {
        background-color: #262730;
        color: white;
        border: 1px solid #4B5563;
        border-radius: 5px;
    }
    .stNumberInput>div>div>input {
        background-color: #262730;
        color: white;
    }
    
    /* Action Button (Gradient) */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        color: #000000;
        font-weight: bold;
        border: none;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        transition: transform 0.2s ease-in-out;
    }
    div.stButton > button:first-child:hover {
        transform: scale(1.02);
        box-shadow: 0px 4px 15px rgba(0, 201, 255, 0.4);
    }

    /* Metric Cards */
    div[data-testid="metric-container"] {
        background-color: #1F2937;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #00C9FF;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    div[data-testid="metric-container"] label {
        color: #9CA3AF; /* Muted label color */
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #F3F4F6; /* Bright value color */
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1F2937;
        border-radius: 5px;
        color: #FAFAFA;
    }
    
    /* Titles */
    h1 {
        background: -webkit-linear-gradient(left, #00C9FF, #92FE9D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    h2, h3 {
        color: #E5E7EB;
    }
    
    /* Footer Credit */
    .footer-credit {
        position: fixed;
        bottom: 10px;
        right: 10px;
        color: #4B5563;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

import yfinance as yf
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF

# Machine Learning Imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Technical Indicators
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator

# Apply Dark Theme to Charts
plt.style.use('dark_background')
# Custom colors: Cyan, Magenta, Yellow, Green, Red
COLORS = ['#00C9FF', '#FF00FF', '#FFD700', '#00FF00', '#FF4B4B']

# --------------------------
# 2. Session State Initialization
# --------------------------
if 'report_data' not in st.session_state:
    st.session_state['report_data'] = None

# --------------------------
# 3. PDF Generator Class (UPDATED)
# --------------------------
class PDFReport(FPDF):
    def header(self):
        # 1. Set Dark Background for every page
        self.set_fill_color(14, 17, 23)  # #0E1117 (Dark Grey)
        self.rect(0, 0, 210, 297, 'F')
        
        # 2. Header Content
        self.set_y(10)
        self.set_font('Arial', 'B', 10)
        self.set_text_color(0, 201, 255)  # Cyan
        self.cell(0, 10, 'AI-Driven Portfolio Management System', 0, 0, 'L')
        self.set_font('Arial', 'I', 9)
        self.cell(0, 10, 'Dev: Advaith Parimisetti', 0, 1, 'R')
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(156, 163, 175)  # Light Grey
        # Updated Footer Credit
        self.cell(0, 10, f'Page {self.page_no()} | Authored by Advaith Parimisetti | CONFIDENTIAL', 0, 0, 'C')

    def create_cover_page(self, ticker):
        self.add_page()
        self.ln(60)
        
        # Title
        self.set_font('Arial', 'B', 32)
        self.set_text_color(255, 255, 255) # White
        self.cell(0, 20, "INVESTMENT MEMORANDUM", 0, 1, 'C')
        
        # Subtitle / Ticker
        self.set_font('Arial', 'B', 24)
        self.set_text_color(0, 201, 255) # Cyan
        self.cell(0, 20, f"Analysis for: {ticker}", 0, 1, 'C')
        
        # Date & Author
        self.ln(10)
        self.set_font('Arial', '', 14)
        self.set_text_color(200, 200, 200)
        self.cell(0, 10, f"Generated: {datetime.date.today().strftime('%Y-%m-%d')}", 0, 1, 'C')
        
        self.ln(5)
        self.set_font('Arial', 'B', 14)
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, "Prepared by: Advaith Parimisetti", 0, 1, 'C')
        
        # Watermark/Note
        self.ln(30)
        self.set_font('Courier', 'B', 12)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, "STRICTLY CONFIDENTIAL", 0, 1, 'C')

    def add_section_title(self, title):
        self.set_font('Arial', 'B', 16)
        self.set_text_color(0, 201, 255) # Cyan
        self.cell(0, 15, title, 0, 1, 'L')
        # Line break (spacer)
        self.set_draw_color(0, 201, 255)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def add_text(self, text):
        self.set_font('Arial', '', 11)
        self.set_text_color(250, 250, 250) # Off-White
        self.multi_cell(0, 6, text)
        self.ln(5)

    def add_table(self, data_dict):
        """Renders a dictionary or simple data as a structured grid."""
        self.set_font('Arial', '', 10)
        self.set_text_color(255, 255, 255)
        self.set_draw_color(75, 85, 99) # Grey border
        self.set_fill_color(31, 41, 55) # Darker cell bg

        # Calculate col widths
        col_w = 45
        val_w = 45
        
        self.ln(2)
        for key, value in data_dict.items():
            # Key Cell
            self.set_font('Arial', 'B', 10)
            self.cell(col_w, 10, str(key), 1, 0, 'L', 1)
            
            # Value Cell
            self.set_font('Arial', '', 10)
            # Format numbers if floats
            if isinstance(value, float):
                display_val = f"{value:.2f}"
            else:
                display_val = str(value)
                
            self.cell(val_w, 10, display_val, 1, 1, 'L', 0) 
        self.ln(5)

    def add_chart(self, fig):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                # Save with dark background for "Dark Mode" PDF look
                fig.savefig(tmpfile.name, format='png', bbox_inches='tight', dpi=150, facecolor=fig.get_facecolor())
                
                # Center the image
                self.image(tmpfile.name, x=15, w=180)
                self.ln(5)
        except Exception:
            pass

# --------------------------
# 4. Helper Functions
# --------------------------
def ensure_series(x):
    if isinstance(x, pd.DataFrame):
        return x.iloc[:, 0] if x.shape[1] >= 1 else pd.Series(dtype=float)
    return x

@st.cache_data(ttl=3600)
def safe_download(ticker, period="1y", interval="1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    except: return None
    if df is None or df.empty: return None
    if isinstance(df.columns, pd.MultiIndex):
        try: df = df.xs(ticker, axis=1, level=1, drop_level=True)
        except: pass
    if "Adj Close" in df.columns: df["Close"] = df["Adj Close"]
    if "Close" not in df.columns: return None
    df["Close"] = ensure_series(df["Close"]).astype(float)
    df = df.dropna()
    return df

def fetch_fundamentals(ticker):
    try:
        info = yf.Ticker(ticker).info
        return {
            "P/E Ratio": info.get("trailingPE", "N/A"),
            "EPS": info.get("trailingEps", "N/A"),
            "ROE": info.get("returnOnEquity", "N/A"),
            "Debt/Equity": info.get("debtToEquity", "N/A"),
            "Beta": info.get("beta", "N/A"),
            "52W High": info.get("fiftyTwoWeekHigh", "N/A"),
            "52W Low": info.get("fiftyTwoWeekLow", "N/A")
        }
    except: return {}

# --- Analysis Logic with Beautified Charts ---
def analyze_ticker(df, ticker):
    df["SMA50"] = ensure_series(df["Close"].rolling(50).mean())
    df["SMA200"] = ensure_series(df["Close"].rolling(200).mean())
    bb = BollingerBands(close=df["Close"], window=20, window_dev=2)
    df["BB_High"] = bb.bollinger_hband()
    df["BB_Low"] = bb.bollinger_lband()
    rsi = RSIIndicator(close=df["Close"], window=14)
    df["RSI"] = rsi.rsi()
    
    # Calculate MACD manually for signal strategy
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    
    df["Signal"] = 0
    mask_buy = (df["SMA50"] > df["SMA200"]) & (df["RSI"] < 70)
    mask_sell = (df["SMA50"] < df["SMA200"]) & (df["RSI"] > 30)
    df.loc[mask_buy, "Signal"] = 1
    df.loc[mask_sell, "Signal"] = -1
    
    # --- CHART 1: Price & Indicators ---
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8), gridspec_kw={"height_ratios":[3,1]}, sharex=True)
    
    # Set dark background to match Streamlit
    fig.patch.set_facecolor('#0E1117')
    ax1.set_facecolor('#0E1117')
    ax2.set_facecolor('#0E1117')
    
    ax1.plot(df.index, df["Close"], label="Close", color=COLORS[0], linewidth=1.5)
    ax1.plot(df.index, df["SMA50"], label="SMA50", color=COLORS[2], linewidth=1, linestyle="--")
    ax1.plot(df.index, df["SMA200"], label="SMA200", color=COLORS[1], linewidth=1, linestyle="--")
    ax1.fill_between(df.index, df["BB_High"], df["BB_Low"], color=COLORS[0], alpha=0.1)
    
    buys = df[df["Signal"]==1]
    sells = df[df["Signal"]==-1]
    if not buys.empty: ax1.scatter(buys.index, buys["Close"], marker="^", color=COLORS[3], s=100, label="Buy Signal", zorder=5)
    if not sells.empty: ax1.scatter(sells.index, sells["Close"], marker="v", color=COLORS[4], s=100, label="Sell Signal", zorder=5)
    
    ax1.set_title(f"{ticker} Price & Indicators", color='white', fontsize=14, fontweight='bold')
    ax1.legend(facecolor='#1F2937', labelcolor='white')
    ax1.grid(True, color='#374151', alpha=0.3)
    ax1.tick_params(axis='x', colors='white')
    ax1.tick_params(axis='y', colors='white')

    # --- CHART 2: RSI ---
    ax2.plot(df.index, df["RSI"], color=COLORS[1], label="RSI")
    ax2.axhline(70, color=COLORS[4], linestyle="--", linewidth=1)
    ax2.axhline(30, color=COLORS[3], linestyle="--", linewidth=1)
    ax2.set_title("RSI Momentum", color='white', fontsize=12)
    ax2.grid(True, color='#374151', alpha=0.3)
    ax2.tick_params(axis='x', colors='white')
    ax2.tick_params(axis='y', colors='white')
    
    plt.tight_layout()
    return df, fig

def run_ml(df, ticker, rf_trees, lstm_epochs):
    data = df.copy().dropna()
    data["Return"] = data["Close"].pct_change()
    data["Target"] = data["Return"].shift(-1)
    for i in range(1, 6):
        data[f"Lag_{i}"] = data["Return"].shift(i)
    data["Vol_20"] = data["Close"].rolling(20).std() / data["Close"]
    data["MACD_Norm"] = data["MACD"] / data["Close"]
    data = data.dropna()
    
    if len(data) < 50: return None, 0, 0, []
    
    feature_cols = [c for c in data.columns if "Lag" in c or "Vol" in c or "MACD" in c]
    X = data[feature_cols].values
    y = data["Target"].values
    
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    base_prices = data["Close"].iloc[split:].values
    actual = base_prices * (1 + y_test)
    
    rf = RandomForestRegressor(n_estimators=rf_trees, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    pred_rf = base_prices * (1 + rf.predict(X_test))
    rmse_rf = math.sqrt(mean_squared_error(actual, pred_rf))
    
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_predictors = [(feature_cols[i], importances[i]) for i in indices[:5]]
    
    scaler = MinMaxScaler((-1,1))
    X_l = scaler.fit_transform(X).reshape(-1, 1, len(feature_cols))
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(1, len(feature_cols))),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_l[:split], y_train, epochs=lstm_epochs, verbose=0)
    pred_lstm = base_prices * (1 + model.predict(X_l[split:], verbose=0).flatten())
    rmse_lstm = math.sqrt(mean_squared_error(actual, pred_lstm))
    
    # --- CHART: Forecast ---
    fig, ax = plt.subplots(figsize=(10,4))
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    
    ax.plot(data.index[split:], actual, label="Actual Price", color='white', alpha=0.6, linewidth=1.5)
    ax.plot(data.index[split:], pred_rf, label=f"RF Pred (RMSE={rmse_rf:.2f})", linestyle="--", color=COLORS[3])
    ax.plot(data.index[split:], pred_lstm, label=f"LSTM Pred (RMSE={rmse_lstm:.2f})", linestyle="--", color=COLORS[2])
    
    ax.set_title(f"{ticker} - Machine Learning Forecast (Test Set)", color='white', fontsize=12)
    ax.legend(facecolor='#1F2937', labelcolor='white')
    ax.grid(True, color='#374151', alpha=0.3)
    ax.tick_params(colors='white')
    
    return fig, rmse_rf, rmse_lstm, top_predictors

def run_monte_carlo(close_series, sims, days, method="gbm", title="Monte Carlo"):
    S0 = close_series.iloc[-1]
    log_rets = np.log(1 + close_series.pct_change().dropna())
    mu, sigma = log_rets.mean(), log_rets.std()
    
    paths = np.zeros((days, sims))
    paths[0] = S0
    rng = np.random.default_rng()
    
    if method == "gbm":
        for t in range(1, days):
            z = rng.normal(0, 1, sims)
            paths[t] = paths[t-1] * np.exp((mu - 0.5*sigma**2) + sigma*z)
    else:
        pool = log_rets.values
        for t in range(1, days):
            r = rng.choice(pool, size=sims)
            paths[t] = paths[t-1] * np.exp(r)
            
    fig, ax = plt.subplots(figsize=(10,5))
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    
    # Plot paths with low opacity
    ax.plot(paths, color=COLORS[0], alpha=0.05)
    ax.set_title(title, color='white', fontsize=12)
    ax.set_xlabel("Days", color='white')
    ax.set_ylabel("Price", color='white')
    ax.grid(True, color='#374151', alpha=0.3)
    ax.tick_params(colors='white')
    
    final_vals = paths[-1]
    fig_hist, ax_h = plt.subplots(figsize=(10,4))
    fig_hist.patch.set_facecolor('#0E1117')
    ax_h.set_facecolor('#0E1117')
    
    ax_h.hist(final_vals, bins=50, alpha=0.8, color=COLORS[1], edgecolor='#0E1117')
    ax_h.set_title(f"{title} - Final Distribution", color='white')
    ax_h.grid(True, color='#374151', alpha=0.3)
    ax_h.tick_params(colors='white')
    
    return fig, fig_hist, final_vals

# --------------------------
# 5. UI Layout (Specific Request)
# --------------------------
# SIDEBAR CREDITS
with st.sidebar:
    st.title("About")
    st.info("Developed by **Advaith Parimisetti**")
    st.markdown("---")
    st.write("This tool uses AI & ML to analyze stock trends and optimize portfolios.")
    st.write("This is not financial advice. Always do your own research before investing.")

st.title("AI-Driven Stock Analysis & Portfolio Management System")
st.markdown("Enter your stock details below to generate a comprehensive report.")

with st.form("analysis_settings"):
    
    # Row 1
    st.subheader("1. Asset Selection")
    col1, col2, col3 = st.columns([1, 2, 2])
    with col1:
        ticker_input = st.text_input("Main Ticker", value="AAPL")
    with col2:
        portfolio_input_str = st.text_input("Portfolio Holdings", value="AAPL, MSFT, GOOG, TSLA, RELIANCE.NS")
    with col3:
        peers_input_str = st.text_input("Competitors (Peers)", value="NVDA, AMZN, INTC")

    # Row 2
    st.subheader("2. Time & Strategy")
    t_col1, t_col2, t_col3 = st.columns(3)
    with t_col1:
        period_val = st.selectbox("Data Period", ["6mo", "1y", "2y", "5y", "max"], index=2)
    with t_col2:
        interval_val = st.selectbox("Candle Interval", ["1d", "1wk", "1mo"], index=0)
    with t_col3:
        rebal_freq = st.selectbox("Rebalance Frequency", ["W", "M", "Q"], index=1)

    # Row 3 (Expander)
    with st.expander("⚙️ Advanced AI & Simulation Settings"):
        adv_c1, adv_c2, adv_c3 = st.columns(3)
        with adv_c1:
            st.markdown("**Monte Carlo**")
            mc_method = st.selectbox("Method", ["gbm", "bootstrap"], index=0)
            mc_sims = st.number_input("Simulations", value=500, step=100)
            mc_horizon = st.number_input("Forecast Days", value=252)
        with adv_c2:
            st.markdown("**AI Hyperparameters**")
            rf_trees = st.number_input("RF Trees", value=100, step=50)
            lstm_epochs = st.number_input("LSTM Epochs", value=10, step=5)
        with adv_c3:
            st.markdown("**Trading Costs**")
            txn_cost = st.number_input("Txn Cost (0.001=0.1%)", value=0.001, format="%.4f", step=0.0001)
            slippage = st.number_input("Slippage (0.0005=0.05%)", value=0.0005, format="%.4f", step=0.0001)
            
    st.markdown("###")
    run_btn = st.form_submit_button("🚀 Run Full Analysis", type="primary", use_container_width=True)

# --------------------------
# 6. Main Execution Logic
# --------------------------
if run_btn:
    # UPDATED: 'items' list allows us to mix tables, text, and figures in order
    report = {'main_ticker': ticker_input, 'items': []}
    
    main_ticker = ticker_input.strip().upper()
    portfolio = [x.strip().upper() for x in portfolio_input_str.split(",") if x.strip()]
    peers = [x.strip().upper() for x in peers_input_str.split(",") if x.strip()]
    
    all_tickers = list(set([main_ticker] + portfolio + peers))
    
    st.info(f"Running analysis for: {all_tickers} (period={period_val}, interval={interval_val})")
    
    data_store = {}
    price_store = {} 
    
    # --- A. Per Ticker Analysis ---
    for t in all_tickers:
        is_focus = (t == main_ticker) or (t in portfolio)
        df = safe_download(t, period_val, interval_val)
        if df is not None:
            data_store[t] = df
            if is_focus:
                st.markdown("---")
                st.header(f"Ticker: {t}")
                
                fund = fetch_fundamentals(t)
                st.write("**Fundamentals (sample):**")
                f_cols = st.columns(4)
                f_keys = list(fund.keys())
                for i, k in enumerate(f_keys):
                    f_cols[i % 4].metric(k, fund[k])
                
                # UPDATED: Store as 'table' type for structured PDF rendering
                report['items'].append({'type': 'section', 'title': f"Ticker Analysis: {t}"})
                report['items'].append({'type': 'table', 'data': fund})
                
                df_tech, fig_tech = analyze_ticker(df, t)
                st.pyplot(fig_tech)
                report['items'].append({'type': 'figure', 'title': f"{t} Price & Indicators", 'fig': fig_tech})
                price_store[t] = df_tech
                
                st.write(f"Training ML models for {t} (Target: Next Day Return)...")
                fig_ml, rmse_rf, rmse_lstm, predictors = run_ml(df_tech, t, rf_trees, lstm_epochs)
                if fig_ml:
                    st.pyplot(fig_ml)
                    col1, col2 = st.columns(2)
                    col1.info(f"[{t}] RF RMSE: {rmse_rf:.4f}")
                    col2.info(f"[{t}] LSTM RMSE: {rmse_lstm:.4f}")
                    
                    st.write(f"**Top 5 Predictors for {t}:**")
                    for p_name, p_val in predictors:
                        st.caption(f"{p_name}: {p_val:.4f}")
                        
                    report['items'].append({'type': 'figure', 'title': f"{t} - Machine Learning Forecast", 'fig': fig_ml})

    # --- B. Peer Comparison ---
    if len(peers) > 0:
        st.markdown("---")
        st.header("Peer Comparison")
        valid_peers = [p for p in peers if p in data_store]
        if main_ticker in data_store and main_ticker not in valid_peers:
            valid_peers.insert(0, main_ticker)
            
        if len(valid_peers) > 1:
            price_df = pd.DataFrame({t: data_store[t]["Close"] for t in valid_peers}).dropna()
            
            peer_recs = []
            for p in valid_peers:
                peer_recs.append({"Ticker": p, **fetch_fundamentals(p)})
            st.write("Peer Fundamentals:")
            st.dataframe(pd.DataFrame(peer_recs).set_index("Ticker"))
            
            rets_df = price_df.pct_change().dropna()
            fig_corr, ax = plt.subplots(figsize=(8,6))
            fig_corr.patch.set_facecolor('#0E1117')
            ax.set_facecolor('#0E1117')
            sns.heatmap(rets_df.corr(), annot=True, cmap="coolwarm", ax=ax, cbar=False)
            ax.tick_params(colors='white')
            ax.set_title("Peer Returns Correlation", color='white')
            st.pyplot(fig_corr)
            report['items'].append({'type': 'section', 'title': "Peer Comparison"})
            report['items'].append({'type': 'figure', 'title': "Peer Returns Correlation", 'fig': fig_corr})
            
            norm_df = price_df / price_df.iloc[0]
            fig_norm, ax = plt.subplots(figsize=(10,5))
            fig_norm.patch.set_facecolor('#0E1117')
            ax.set_facecolor('#0E1117')
            for c in norm_df.columns:
                ax.plot(norm_df.index, norm_df[c], label=c)
            ax.set_title("Relative Performance (Normalized to 1.0)", color='white')
            ax.legend(facecolor='#1F2937', labelcolor='white')
            ax.grid(True, color='#374151', alpha=0.3)
            ax.tick_params(colors='white')
            st.pyplot(fig_norm)
            report['items'].append({'type': 'figure', 'title': "Relative Performance", 'fig': fig_norm})

    # --- C. Portfolio ---
    st.markdown("---")
    st.header("Portfolio Analysis & Strategy")
    valid_port = [p for p in portfolio if p in data_store]
    
    if len(valid_port) > 1:
        port_prices = pd.DataFrame({t: data_store[t]["Close"] for t in valid_port}).dropna()
        port_rets = port_prices.pct_change().dropna()
        
        mu = port_rets.mean()
        cov = port_rets.cov()
        try:
            inv_cov = np.linalg.pinv(cov.values)
            ones = np.ones(len(mu))
            w = inv_cov @ mu.values / (ones @ inv_cov @ mu.values)
            w = np.clip(w, 0, None)
            w = w / w.sum()
        except:
            w = np.repeat(1/len(valid_port), len(valid_port))
        
        st.subheader("Optimized weights (non-negative mean-variance):")
        w_df = pd.DataFrame({"Asset": port_rets.columns, "Weight": w})
        st.dataframe(w_df.style.format({"Weight": "{:.3%}"}), use_container_width=True)
        
        # UPDATED: Store weights as table for PDF
        weight_dict = dict(zip(port_rets.columns, [round(x, 4) for x in w]))
        report['items'].append({'type': 'section', 'title': "Portfolio Optimization"})
        report['items'].append({'type': 'table', 'data': weight_dict})
        
        daily_ret = (port_rets * w).sum(axis=1)
        equity = (1 + daily_ret).cumprod()
        drawdown = equity / equity.cummax() - 1
        
        sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252)
        st.success(f"Portfolio Backtest: Sharpe = {sharpe:.3f}, Final Equity = {equity.iloc[-1]:.3f}, Max Drawdown = {drawdown.min():.2%}")
        
        # Portfolio Equity Curve
        fig_eq, ax = plt.subplots(figsize=(10,4))
        fig_eq.patch.set_facecolor('#0E1117')
        ax.set_facecolor('#0E1117')
        ax.plot(equity.index, equity, color=COLORS[0])
        ax.set_title("Portfolio Equity Curve", color='white')
        ax.grid(True, color='#374151', alpha=0.3)
        ax.tick_params(colors='white')
        st.pyplot(fig_eq)
        report['items'].append({'type': 'figure', 'title': "Portfolio Equity Curve", 'fig': fig_eq})
        
        # Drawdown
        fig_dd, ax = plt.subplots(figsize=(10,4))
        fig_dd.patch.set_facecolor('#0E1117')
        ax.set_facecolor('#0E1117')
        ax.plot(drawdown.index, drawdown, color=COLORS[4])
        ax.set_title("Portfolio Drawdown", color='white')
        ax.grid(True, color='#374151', alpha=0.3)
        ax.tick_params(colors='white')
        st.pyplot(fig_dd)
        report['items'].append({'type': 'figure', 'title': "Portfolio Drawdown", 'fig': fig_dd})
        
        # Signal Strategy
        st.subheader("Signal-Weighted Strategy")
        sig_dict = {}
        for t in valid_port:
            if t in price_store:
                sig_dict[t] = price_store[t]["Signal"]
            else:
                sig_dict[t] = pd.Series(0, index=port_rets.index)
        
        sig_df = pd.DataFrame(sig_dict).reindex(port_rets.index).fillna(0).shift(1)
        vol = port_rets.rolling(60).std().bfill() + 1e-6
        risk_adj_sig = sig_df / vol
        w_strat = risk_adj_sig.div(risk_adj_sig.abs().sum(axis=1).replace(0,1), axis=0).fillna(0)
        
        turnover = w_strat.diff().abs().sum(axis=1)
        strat_ret = (port_rets * w_strat).sum(axis=1) - (turnover * (txn_cost + slippage))
        strat_cum = (1 + strat_ret).cumprod()
        
        ann_ret_strat = strat_ret.mean() * 252
        vol_strat = strat_ret.std() * np.sqrt(252)
        sharpe_strat = ann_ret_strat / vol_strat if vol_strat != 0 else 0
        
        st.info(f"Signal-weighted strategy: Annual Return {ann_ret_strat:.2%}, Vol {vol_strat:.2%}, Sharpe {sharpe_strat:.3f}")
        
        fig_strat, ax = plt.subplots(figsize=(10,4))
        fig_strat.patch.set_facecolor('#0E1117')
        ax.set_facecolor('#0E1117')
        ax.plot(strat_cum.index, strat_cum, color=COLORS[2])
        ax.set_title("Signal-Weighted Strategy (Cumulative)", color='white')
        ax.grid(True, color='#374151', alpha=0.3)
        ax.tick_params(colors='white')
        st.pyplot(fig_strat)
        report['items'].append({'type': 'figure', 'title': "Signal-Weighted Strategy", 'fig': fig_strat})
        
        # Rebalancing
        st.subheader(f"Rebalancing Simulation ({rebal_freq})")
        reb_rets_series = (port_rets * w).sum(axis=1)
        reb_cum = (1 + reb_rets_series).cumprod()
        
        ann_reb = reb_rets_series.mean() * 252
        vol_reb = reb_rets_series.std() * np.sqrt(252)
        sharpe_reb = ann_reb / vol_reb
        
        st.write(f"**Rebalanced ({rebal_freq}):** Annual Return {ann_reb:.2%}, Vol {vol_reb:.2%}, Sharpe {sharpe_reb:.3f}")
        
        fig_reb, ax = plt.subplots(figsize=(10,4))
        fig_reb.patch.set_facecolor('#0E1117')
        ax.set_facecolor('#0E1117')
        ax.plot(reb_cum.index, reb_cum, color=COLORS[3])
        ax.set_title(f"Rebalanced ({rebal_freq}) Cumulative Return", color='white')
        ax.grid(True, color='#374151', alpha=0.3)
        ax.tick_params(colors='white')
        st.pyplot(fig_reb)
        report['items'].append({'type': 'figure', 'title': f"Rebalanced ({rebal_freq}) Strategy", 'fig': fig_reb})
        
        # MC Strategy
        if not strat_cum.empty:
            st.subheader("Monte Carlo: Signal Strategy")
            fig_mc_strat, fig_hist_strat, final_strat = run_monte_carlo(strat_cum, mc_sims, mc_horizon, mc_method, "Strategy MC Simulation")
            col1, col2 = st.columns(2)
            with col1: st.pyplot(fig_mc_strat)
            with col2: st.pyplot(fig_hist_strat)
            st.write(f"Median final: {np.median(final_strat):.2f}, 5%: {np.percentile(final_strat,5):.2f}, 95%: {np.percentile(final_strat,95):.2f}")
            report['items'].append({'type': 'figure', 'title': "Strategy Monte Carlo", 'fig': fig_mc_strat})
            report['items'].append({'type': 'figure', 'title': "Strategy MC Distribution", 'fig': fig_hist_strat})

    # --- D. MC Main Ticker ---
    if main_ticker in data_store:
        st.markdown("---")
        st.header(f"Monte Carlo (Price-level): {main_ticker}")
        close_series = data_store[main_ticker]["Close"]
        
        log_rets = np.log(1 + close_series.pct_change().dropna())
        mu_p, sigma_p = log_rets.mean(), log_rets.std()
        last_p = close_series.iloc[-1]
        st.caption(f"{main_ticker} mu={mu_p:.6f}, sigma={sigma_p:.6f}, last_price={last_p:.2f}")
        
        fig_mc_price, fig_hist_price, final_vals = run_monte_carlo(close_series, mc_sims, mc_horizon, mc_method, f"{main_ticker} Price MC")
        
        col1, col2 = st.columns(2)
        with col1: st.pyplot(fig_mc_price)
        with col2: st.pyplot(fig_hist_price)
        
        st.success(f"Final Price Projections (95% CI): {np.percentile(final_vals, 5):.2f} — {np.percentile(final_vals, 95):.2f}")
        
        report['items'].append({'type': 'figure', 'title': f"{main_ticker} Monte Carlo Paths", 'fig': fig_mc_price})
        report['items'].append({'type': 'figure', 'title': f"{main_ticker} Monte Carlo Distribution", 'fig': fig_hist_price})

    st.session_state['report_data'] = report
    st.success("Done. Scroll up for charts and tables.")

# --------------------------
# 7. PDF Download
# --------------------------
if st.session_state['report_data'] is not None:
    st.markdown("---")
    st.subheader("📥 Export Results")
    
    col_d1, col_d2 = st.columns([1, 4])
    with col_d1:
        if st.button("Generate PDF Report"):
            data = st.session_state['report_data']
            pdf = PDFReport()
            
            # 1. Cover Page
            pdf.create_cover_page(data['main_ticker'])
            
            # 2. Iterate through items
            for item in data['items']:
                if item['type'] == 'section':
                    pdf.add_page()
                    pdf.add_section_title(item['title'])
                    
                elif item['type'] == 'table':
                    pdf.add_text("The following table:")
                    pdf.add_table(item['data'])
                    
                elif item['type'] == 'text':
                    pdf.add_text(item['content'])
                    
                elif item['type'] == 'figure':
                    # Ensure charts are on a page where they fit or add new page
                    # For simplicity, we just add the chart. 
                    # If we want 1 chart per page, we can uncomment pdf.add_page()
                    # pdf.add_page()
                    pdf.add_chart(item['fig'])
            
            pdf_out = pdf.output(dest='S').encode('latin-1')
            st.download_button(
                label="Download PDF",
                data=pdf_out,
                file_name=f"{data['main_ticker']}_Analysis.pdf",
                mime="application/pdf"
            )