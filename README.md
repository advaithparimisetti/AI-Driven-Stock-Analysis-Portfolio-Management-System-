# 📈 AI-Driven Stock Analysis & Portfolio Management System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)](https://www.tensorflow.org/)

https://ai-driven-stock-analysis-portfolio-management-system.streamlit.app/

## 🚀 Overview
This project is a sophisticated financial dashboard built with **Streamlit** that leverages Artificial Intelligence and Machine Learning to analyze stock trends, optimize portfolios, and simulate trading strategies. 

Designed for quantitative analysis, the tool combines technical indicators with **Random Forest** and **LSTM (Long Short-Term Memory)** deep learning models to forecast price movements. It also includes a robust PDF generation engine to produce professional "Investment Memorandums."

## ✨ Key Features

### 🧠 AI & Machine Learning
* **Price Forecasting:** Utilizes **Random Forest Regressors** and **LSTM Neural Networks** to predict next-day returns based on lagged features and volatility.
* **Feature Importance:** Identifies top technical indicators driving price movements.

### 📊 Technical Analysis
* **Indicators:** Automated calculation of SMA (50/200), Bollinger Bands, RSI, and MACD.
* **Signal Generation:** Algorithmic Buy/Sell signals based on Moving Average crossovers and RSI thresholds.

### 💼 Portfolio Management
* **Optimization:** Calculates optimal asset weights using inverse covariance (Mean-Variance style).
* **Backtesting:** analyzes cumulative equity, maximum drawdown, and Sharpe ratios.
* **Rebalancing:** Simulates Weekly, Monthly, or Quarterly portfolio rebalancing strategies.

### 🎲 Advanced Simulations
* **Monte Carlo Simulation:** projects future price paths using Geometric Brownian Motion (GBM) or Bootstrap methods.
* **Strategy Simulation:** Stress-tests algorithmic trading strategies over thousands of potential market scenarios.

### 📑 Automated Reporting
* **PDF Export:** Generates a dark-themed, multi-page **Investment Memorandum** containing:
    * Cover Page
    * Fundamental Analysis Tables
    * High-res Charts (Technical, ML Forecasts, Peer Comparison)
    * Portfolio Weights & Performance Metrics

## 🛠️ Tech Stack
* **UI/UX:** Streamlit (Custom CSS for Dark Mode)
* **Data Source:** `yfinance`
* **Data Processing:** `pandas`, `numpy`
* **Visualization:** `matplotlib`, `seaborn`
* **Machine Learning:** `scikit-learn` (Random Forest), `tensorflow`/`keras` (LSTM)
* **Technical Analysis:** `ta` library
* **Reporting:** `fpdf`

## ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/ai-stock-analysis.git](https://github.com/yourusername/ai-stock-analysis.git)
    cd ai-stock-analysis
    ```

2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

## 📦 Requirements
Ensure your `requirements.txt` contains the following:
```text
streamlit
yfinance
pandas
numpy
matplotlib
seaborn
fpdf
scikit-learn
tensorflow
ta
