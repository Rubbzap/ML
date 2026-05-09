from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from backtesting import run_directional_backtest
from config import DEFAULT_TICKER, RANDOM_STATE
from data_loader import download_stock_data
from features import add_technical_indicators, get_feature_columns
from modeling import train_and_evaluate


st.set_page_config(
    page_title="Stock Movement Forecasting",
    page_icon="chart",
    layout="wide",
    initial_sidebar_state="expanded",
)


POPULAR_TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL", "AMZN", "META", "SPY", "QQQ"]
RANGE_KEYS = ["1D", "5D", "1M", "3M", "YTD", "1Y", "5Y", "10Y", "ALL"]
MIN_TRAIN_ROWS = 180


TEXT = {
    "TH": {
        "app_title": "Stock Movement Forecasting",
        "app_subtitle": "แดชบอร์ดพยากรณ์ทิศทางหุ้นด้วย Machine Learning",
        "language": "ภาษา / Language",
        "control_panel": "ตั้งค่าการวิเคราะห์",
        "quick_pick": "เลือกหุ้นยอดนิยม",
        "custom_ticker": "ใช้สัญลักษณ์ที่พิมพ์เอง",
        "ticker": "สัญลักษณ์หุ้น",
        "range": "ช่วงเวลา",
        "model": "โมเดล",
        "tune": "ปรับจูนโมเดล",
        "run": "วิเคราะห์หุ้น",
        "ready": "เลือกหุ้น ช่วงเวลา และกด วิเคราะห์หุ้น เพื่อเริ่มใช้งาน",
        "loading": "กำลังดึงข้อมูล ฝึกโมเดล และเตรียมกราฟ...",
        "signal": "สัญญาณล่าสุด",
        "buy": "BUY",
        "sell": "SELL / AVOID",
        "confidence": "โอกาสขึ้น",
        "last_close": "ราคาปิดล่าสุด",
        "accuracy": "ความแม่นยำ",
        "f1": "F1-score",
        "strategy_return": "ผลตอบแทนกลยุทธ์",
        "price_tab": "กราฟราคา",
        "model_tab": "ผลโมเดล",
        "backtest_tab": "Backtest",
        "feature_tab": "ตัวแปรสำคัญ",
        "explain_tab": "อธิบายตัวแปร",
        "candlestick": "กราฟราคาและปริมาณซื้อขาย",
        "metrics": "ตัวชี้วัดโมเดล",
        "confusion": "Confusion Matrix",
        "best_params": "ค่าพารามิเตอร์ที่ดีที่สุด",
        "equity_curve": "Equity Curve",
        "backtest_metrics": "ตัวชี้วัด Backtest",
        "feature_importance": "ตัวแปรที่โมเดลให้ความสำคัญ",
        "no_importance": "โมเดลนี้ไม่สามารถแสดง feature importance ได้",
        "rows": "จำนวนข้อมูลในกราฟ",
        "train_rows": "จำนวนข้อมูลฝึกโมเดล",
        "period": "เริ่มต้น",
        "end_date": "สิ้นสุด",
        "disclaimer": "ใช้เพื่อการศึกษาเท่านั้น ไม่ใช่คำแนะนำการลงทุน",
        "actual_down": "จริง: ลง",
        "actual_up": "จริง: ขึ้น",
        "pred_down": "ทำนาย: ลง",
        "pred_up": "ทำนาย: ขึ้น",
        "range_note": "ช่วงเวลาใช้ควบคุมกราฟ ส่วนโมเดลยังใช้ข้อมูลย้อนหลังที่มากพอเพื่อให้ประเมินผลได้เสถียร",
        "feature_name": "ตัวแปร",
        "importance": "ความสำคัญ",
        "plain_meaning": "ความหมายแบบง่าย",
        "how_to_read": "อ่านค่านี้ยังไง",
        "top_feature_help": "ตารางนี้ช่วยบอกว่าโมเดลอาศัยข้อมูลแบบไหนมากที่สุด เช่น momentum, volatility หรือ volume",
        "not_enough": "ข้อมูลไม่พอสำหรับฝึกโมเดล ลองเลือกหุ้นที่มีประวัติยาวขึ้น",
        "model_names": {
            "logistic": "Logistic Regression",
            "random_forest": "Random Forest",
            "xgboost": "XGBoost",
        },
    },
    "EN": {
        "app_title": "Stock Movement Forecasting",
        "app_subtitle": "Machine learning dashboard for stock direction signals",
        "language": "Language",
        "control_panel": "Analysis Settings",
        "quick_pick": "Popular tickers",
        "custom_ticker": "Use typed ticker",
        "ticker": "Ticker",
        "range": "Range",
        "model": "Model",
        "tune": "Tune model",
        "run": "Analyze stock",
        "ready": "Choose a stock, range, and click Analyze stock to begin",
        "loading": "Downloading data, training model, and preparing charts...",
        "signal": "Latest signal",
        "buy": "BUY",
        "sell": "SELL / AVOID",
        "confidence": "Up probability",
        "last_close": "Latest close",
        "accuracy": "Accuracy",
        "f1": "F1-score",
        "strategy_return": "Strategy return",
        "price_tab": "Price Chart",
        "model_tab": "Model Results",
        "backtest_tab": "Backtest",
        "feature_tab": "Important Variables",
        "explain_tab": "Variable Guide",
        "candlestick": "Price and Volume",
        "metrics": "Model Metrics",
        "confusion": "Confusion Matrix",
        "best_params": "Best Parameters",
        "equity_curve": "Equity Curve",
        "backtest_metrics": "Backtest Metrics",
        "feature_importance": "Variables the model used most",
        "no_importance": "This model does not expose feature importance",
        "rows": "Chart rows",
        "train_rows": "Training rows",
        "period": "Start",
        "end_date": "End",
        "disclaimer": "For education only. This is not investment advice.",
        "actual_down": "Actual Down",
        "actual_up": "Actual Up",
        "pred_down": "Pred Down",
        "pred_up": "Pred Up",
        "range_note": "The range controls the chart. The model still uses enough history for more stable evaluation.",
        "feature_name": "Variable",
        "importance": "Importance",
        "plain_meaning": "Plain meaning",
        "how_to_read": "How to read it",
        "top_feature_help": "This table shows whether the model relies more on momentum, volatility, volume, or trend information.",
        "not_enough": "Not enough data to train the model. Try a stock with a longer history.",
        "model_names": {
            "logistic": "Logistic Regression",
            "random_forest": "Random Forest",
            "xgboost": "XGBoost",
        },
    },
}


RANGE_LABELS = {
    "TH": {
        "1D": "1 วัน",
        "5D": "5 วัน",
        "1M": "1 เดือน",
        "3M": "3 เดือน",
        "YTD": "ตั้งแต่ต้นปี",
        "1Y": "1 ปี",
        "5Y": "5 ปี",
        "10Y": "10 ปี",
        "ALL": "ทั้งหมด",
    },
    "EN": {
        "1D": "1D",
        "5D": "5D",
        "1M": "1M",
        "3M": "3M",
        "YTD": "Year to date",
        "1Y": "1Y",
        "5Y": "5Y",
        "10Y": "10Y",
        "ALL": "All",
    },
}


FEATURE_EXPLANATIONS = {
    "TH": {
        "return_1d": ("ผลตอบแทนวันล่าสุด", "ค่าบวกแปลว่าราคาปิดสูงกว่าวันก่อนหน้า ค่าลบแปลว่าราคาลง"),
        "log_return_1d": ("ผลตอบแทนแบบ log", "ใช้วัดการเปลี่ยนแปลงราคาแบบต่อเนื่อง เหมาะกับงาน time series"),
        "volume_change": ("การเปลี่ยนแปลงของปริมาณซื้อขาย", "ค่าสูงแปลว่าวันนั้นมีแรงซื้อขายมากขึ้นกว่าก่อนหน้า"),
        "rolling_volatility_10": ("ความผันผวนย้อนหลัง 10 วัน", "ยิ่งสูงยิ่งแปลว่าราคาแกว่งแรงในระยะสั้น"),
        "rolling_volatility_20": ("ความผันผวนย้อนหลัง 20 วัน", "ใช้ดูความเสี่ยงและความไม่นิ่งของราคาในช่วงประมาณ 1 เดือน"),
        "rsi_14": ("RSI 14 วัน", "มากกว่า 70 มักแปลว่าซื้อเยอะเกินไป ต่ำกว่า 30 มักแปลว่าขายเยอะเกินไป"),
        "macd": ("MACD", "ใช้ดู momentum ของแนวโน้ม ถ้าค่าสูงขึ้นแปลว่าแรงขาขึ้นเริ่มเด่น"),
        "macd_signal": ("เส้น signal ของ MACD", "ใช้เทียบกับ MACD เพื่อดูจังหวะเปลี่ยนแนวโน้ม"),
        "macd_hist": ("ส่วนต่าง MACD กับ signal", "ค่าสูงขึ้นแปลว่า momentum ขาขึ้นแรงขึ้น"),
        "bb_position": ("ตำแหน่งราคาใน Bollinger Bands", "ใกล้ 1 คือราคาอยู่ใกล้กรอบบน ใกล้ 0 คืออยู่ใกล้กรอบล่าง"),
    },
    "EN": {
        "return_1d": ("Latest daily return", "Positive means the close was higher than the prior day; negative means it fell"),
        "log_return_1d": ("Log return", "A continuous-style price change often used in time series modeling"),
        "volume_change": ("Volume change", "High values mean trading activity increased versus the prior day"),
        "rolling_volatility_10": ("10-day volatility", "Higher values mean the stock has been moving more sharply in the short term"),
        "rolling_volatility_20": ("20-day volatility", "A rough one-month view of risk and price instability"),
        "rsi_14": ("14-day RSI", "Above 70 often suggests overbought; below 30 often suggests oversold"),
        "macd": ("MACD", "A momentum indicator; rising values suggest upward momentum is strengthening"),
        "macd_signal": ("MACD signal line", "Compared with MACD to spot trend changes"),
        "macd_hist": ("MACD histogram", "The gap between MACD and signal; rising values suggest stronger upward momentum"),
        "bb_position": ("Bollinger Band position", "Near 1 means price is near the upper band; near 0 means near the lower band"),
    },
}


@st.cache_data(show_spinner=False, ttl=3600)
def load_data(ticker: str, start: str) -> pd.DataFrame:
    raw = download_stock_data(ticker, start, None)
    return add_technical_indicators(raw)


def t(key: str, lang: str) -> str:
    return TEXT[lang][key]


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #0f172a;
            --panel: #111827;
            --text: #e5e7eb;
            --muted: #94a3b8;
            --line: #334155;
            --green: #22c55e;
            --red: #ef4444;
            --amber: #f59e0b;
            --blue: #38bdf8;
        }
        .stApp {
            background: var(--bg);
            color: var(--text);
        }
        section[data-testid="stSidebar"] {
            background: #0b1120;
            border-right: 1px solid var(--line);
        }
        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 2rem;
            max-width: 1500px;
        }
        h1, h2, h3, h4, label, .stMarkdown, .stMetric {
            color: var(--text) !important;
        }
        button, input, textarea, [data-baseweb="select"] {
            transition: border-color 140ms ease, box-shadow 140ms ease, transform 140ms ease;
        }
        div[data-testid="stMetric"] {
            background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 14px 16px;
        }
        div[data-testid="stMetricLabel"] {
            color: var(--muted);
        }
        div[data-testid="stMetricValue"] {
            color: var(--text);
            font-size: 1.35rem;
        }
        .tv-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            gap: 16px;
            padding: 12px 0 12px;
            border-bottom: 1px solid var(--line);
            margin-bottom: 14px;
        }
        .tv-title {
            font-size: 1.7rem;
            font-weight: 700;
            line-height: 1.2;
        }
        .tv-subtitle {
            color: var(--muted);
            margin-top: 4px;
            font-size: 0.95rem;
        }
        .status-pill {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 132px;
            padding: 9px 12px;
            border-radius: 999px;
            font-weight: 700;
            letter-spacing: 0;
        }
        .status-buy {
            color: #052e16;
            background: var(--green);
        }
        .status-sell {
            color: #450a0a;
            background: var(--red);
        }
        .fine-print {
            color: var(--muted);
            font-size: 0.85rem;
            margin-top: 8px;
        }
        .range-note {
            color: var(--muted);
            font-size: 0.9rem;
            padding: 8px 0 2px;
        }
        div[data-testid="stTabs"] button {
            color: var(--muted);
        }
        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: var(--blue);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def download_start_for_range(range_key: str) -> str:
    if range_key == "ALL":
        return "1900-01-01"
    latest_reasonable_start = pd.Timestamp.today().normalize() - pd.DateOffset(years=11)
    return str(latest_reasonable_start.date())


def filter_visible_range(dataset: pd.DataFrame, range_key: str) -> pd.DataFrame:
    if dataset.empty:
        return dataset

    latest_date = pd.to_datetime(dataset["Date"]).max()
    if range_key == "1D":
        return dataset.tail(1)
    if range_key == "5D":
        return dataset.tail(5)
    if range_key == "1M":
        start = latest_date - pd.DateOffset(months=1)
    elif range_key == "3M":
        start = latest_date - pd.DateOffset(months=3)
    elif range_key == "YTD":
        start = pd.Timestamp(year=latest_date.year, month=1, day=1)
    elif range_key == "1Y":
        start = latest_date - pd.DateOffset(years=1)
    elif range_key == "5Y":
        start = latest_date - pd.DateOffset(years=5)
    elif range_key == "10Y":
        start = latest_date - pd.DateOffset(years=10)
    else:
        return dataset

    visible = dataset[pd.to_datetime(dataset["Date"]) >= start]
    return visible if not visible.empty else dataset.tail(1)


def get_feature_explanation(feature: str, lang: str) -> tuple[str, str]:
    direct = FEATURE_EXPLANATIONS[lang].get(feature)
    if direct:
        return direct

    if feature.startswith("return_lag_"):
        days = feature.replace("return_lag_", "")
        if lang == "TH":
            return (f"ผลตอบแทนย้อนหลัง {days} วัน", "ช่วยให้โมเดลดูว่าก่อนหน้านี้ราคามี momentum ขึ้นหรือลง")
        return (f"Return from {days} day(s) ago", "Helps the model read recent upward or downward momentum")

    if feature.startswith("ma_"):
        days = feature.replace("ma_", "")
        if lang == "TH":
            return (f"ราคาเฉลี่ยย้อนหลัง {days} วัน", "ใช้ดูแนวโน้มราคาโดยลด noise รายวัน")
        return (f"{days}-day moving average", "Shows the broader trend by smoothing daily noise")

    if feature.startswith("price_to_ma_"):
        days = feature.replace("price_to_ma_", "")
        if lang == "TH":
            return (f"ราคาปัจจุบันเทียบกับ MA {days} วัน", "ค่าบวกแปลว่าราคาอยู่เหนือเส้นเฉลี่ย ค่าลบแปลว่าอยู่ต่ำกว่า")
        return (f"Price versus {days}-day MA", "Positive means price is above the average; negative means below it")

    if feature.startswith("bb_"):
        if lang == "TH":
            return ("Bollinger Bands", "ใช้ดูว่าราคาอยู่สูงหรือต่ำเมื่อเทียบกับกรอบความผันผวน")
        return ("Bollinger Bands", "Shows where price sits relative to a volatility band")

    if lang == "TH":
        return ("ตัวแปรเชิงเทคนิค", "เป็นข้อมูลที่ช่วยให้โมเดลจับแนวโน้ม ความผันผวน หรือแรงซื้อขาย")
    return ("Technical variable", "Helps the model capture trend, volatility, or trading activity")


def build_feature_table(importance: pd.DataFrame, lang: str) -> pd.DataFrame:
    rows = []
    for _, row in importance.iterrows():
        meaning, reading = get_feature_explanation(row["feature"], lang)
        rows.append(
            {
                t("feature_name", lang): row["feature"],
                t("importance", lang): round(float(row["importance"]), 4),
                t("plain_meaning", lang): meaning,
                t("how_to_read", lang): reading,
            }
        )
    return pd.DataFrame(rows)


def build_price_chart(dataset: pd.DataFrame, ticker: str, range_key: str, lang: str) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.72, 0.28],
    )
    fig.add_trace(
        go.Candlestick(
            x=dataset["Date"],
            open=dataset["Open"],
            high=dataset["High"],
            low=dataset["Low"],
            close=dataset["Close"],
            name=ticker.upper(),
            increasing_line_color="#22c55e",
            decreasing_line_color="#ef4444",
        ),
        row=1,
        col=1,
    )
    if len(dataset) >= 20:
        fig.add_trace(
            go.Scatter(
                x=dataset["Date"],
                y=dataset["ma_20"],
                mode="lines",
                line=dict(color="#38bdf8", width=1.4),
                name="MA20",
            ),
            row=1,
            col=1,
        )
    if len(dataset) >= 50:
        fig.add_trace(
            go.Scatter(
                x=dataset["Date"],
                y=dataset["ma_50"],
                mode="lines",
                line=dict(color="#f59e0b", width=1.4),
                name="MA50",
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Bar(
            x=dataset["Date"],
            y=dataset["Volume"],
            marker_color="#475569",
            name="Volume",
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        title=f"{ticker.upper()} - {RANGE_LABELS[lang][range_key]}",
        template="plotly_dark",
        height=610,
        margin=dict(l=18, r=18, t=48, b=18),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#111827",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        uirevision=f"{ticker}-{range_key}",
    )
    fig.update_yaxes(gridcolor="#334155", zerolinecolor="#334155")
    fig.update_xaxes(gridcolor="#334155", zerolinecolor="#334155")
    return fig


def build_equity_chart(backtest_df: pd.DataFrame, lang: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=backtest_df["Date"],
            y=backtest_df["market_equity"],
            mode="lines",
            name="Buy and Hold",
            line=dict(color="#94a3b8", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=backtest_df["Date"],
            y=backtest_df["strategy_equity"],
            mode="lines",
            name="ML Strategy",
            line=dict(color="#38bdf8", width=2),
        )
    )
    fig.update_layout(
        title=t("equity_curve", lang),
        template="plotly_dark",
        height=470,
        margin=dict(l=18, r=18, t=48, b=18),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#111827",
        hovermode="x unified",
    )
    fig.update_yaxes(gridcolor="#334155", zerolinecolor="#334155")
    fig.update_xaxes(gridcolor="#334155", zerolinecolor="#334155")
    return fig


def format_metric_table(metrics: dict) -> pd.DataFrame:
    rows = []
    for key, value in metrics.items():
        if key in {"confusion_matrix", "classification_report", "best_params"}:
            continue
        rows.append({"metric": key, "value": round(value, 4)})
    return pd.DataFrame(rows)


def run_analysis(ticker: str, range_key: str, model_name: str, tune: bool) -> dict:
    dataset = load_data(ticker, download_start_for_range(range_key))
    if len(dataset) < MIN_TRAIN_ROWS:
        raise ValueError("not_enough")

    feature_columns = get_feature_columns(dataset)
    result = train_and_evaluate(
        dataset,
        feature_columns,
        model_name,
        tune=tune,
        random_state=RANDOM_STATE,
    )
    backtest_df, backtest_metrics = run_directional_backtest(result.predictions)
    return {
        "ticker": ticker,
        "range_key": range_key,
        "model_name": model_name,
        "tune": tune,
        "dataset": dataset,
        "visible_dataset": filter_visible_range(dataset, range_key),
        "result": result,
        "backtest_df": backtest_df,
        "backtest_metrics": backtest_metrics,
    }


def render_header(ticker: str, lang: str) -> None:
    st.markdown(
        f"""
        <div class="tv-header">
            <div>
                <div class="tv-title">{t('app_title', lang)}</div>
                <div class="tv-subtitle">{t('app_subtitle', lang)}</div>
            </div>
            <div class="tv-subtitle">{ticker.upper()}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state(ticker: str, selected_model_label: str, range_key: str, tune: bool, lang: str) -> None:
    st.info(t("ready", lang))
    cols = st.columns(4)
    cols[0].metric(t("ticker", lang), ticker.upper())
    cols[1].metric(t("range", lang), RANGE_LABELS[lang][range_key])
    cols[2].metric(t("model", lang), selected_model_label)
    cols[3].metric(t("tune", lang), "ON" if tune else "OFF")


def render_results(state: dict, lang: str) -> None:
    ticker = state["ticker"]
    range_key = state["range_key"]
    dataset = state["dataset"]
    visible_dataset = state["visible_dataset"]
    result = state["result"]
    backtest_df = state["backtest_df"]
    backtest_metrics = state["backtest_metrics"]

    latest = result.predictions.iloc[-1]
    signal_is_buy = latest["prediction"] == 1
    signal = t("buy", lang) if signal_is_buy else t("sell", lang)
    signal_class = "status-buy" if signal_is_buy else "status-sell"
    latest_close = dataset.iloc[-1]["Close"]

    top_left, top_right = st.columns([1.15, 4.85])
    with top_left:
        st.markdown(
            f"""
            <div class="status-pill {signal_class}">{signal}</div>
            <div class="fine-print">{t('signal', lang)}</div>
            """,
            unsafe_allow_html=True,
        )
    with top_right:
        cols = st.columns(5)
        cols[0].metric(t("last_close", lang), f"${latest_close:,.2f}")
        cols[1].metric(t("confidence", lang), f"{latest['probability_up']:.1%}")
        cols[2].metric(t("accuracy", lang), f"{result.metrics['accuracy']:.3f}")
        cols[3].metric(t("f1", lang), f"{result.metrics['f1']:.3f}")
        cols[4].metric(t("strategy_return", lang), f"{backtest_metrics['cumulative_strategy_return']:.2%}")

    st.markdown(f"<div class='range-note'>{t('range_note', lang)}</div>", unsafe_allow_html=True)

    tab_price, tab_metrics, tab_backtest, tab_features, tab_explain = st.tabs(
        [
            t("price_tab", lang),
            t("model_tab", lang),
            t("backtest_tab", lang),
            t("feature_tab", lang),
            t("explain_tab", lang),
        ]
    )

    chart_config = {
        "displaylogo": False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        "responsive": True,
    }

    with tab_price:
        st.plotly_chart(
            build_price_chart(visible_dataset, ticker, range_key, lang),
            use_container_width=True,
            config=chart_config,
        )
        info_cols = st.columns(4)
        info_cols[0].metric(t("rows", lang), f"{len(visible_dataset):,}")
        info_cols[1].metric(t("train_rows", lang), f"{len(dataset):,}")
        info_cols[2].metric(t("period", lang), str(visible_dataset["Date"].min().date()))
        info_cols[3].metric(t("end_date", lang), str(visible_dataset["Date"].max().date()))

    with tab_metrics:
        left, right = st.columns([1.05, 1])
        with left:
            st.subheader(t("metrics", lang))
            st.dataframe(format_metric_table(result.metrics), use_container_width=True, hide_index=True)
            if result.metrics.get("best_params"):
                st.subheader(t("best_params", lang))
                st.json(result.metrics["best_params"])
        with right:
            st.subheader(t("confusion", lang))
            cm = pd.DataFrame(
                result.metrics["confusion_matrix"],
                index=[t("actual_down", lang), t("actual_up", lang)],
                columns=[t("pred_down", lang), t("pred_up", lang)],
            )
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", template="plotly_dark")
            fig_cm.update_layout(
                paper_bgcolor="#0f172a",
                plot_bgcolor="#111827",
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig_cm, use_container_width=True, config=chart_config)

    with tab_backtest:
        st.plotly_chart(build_equity_chart(backtest_df, lang), use_container_width=True, config=chart_config)
        st.subheader(t("backtest_metrics", lang))
        st.dataframe(
            pd.DataFrame(
                [{"metric": key, "value": round(value, 4)} for key, value in backtest_metrics.items()]
            ),
            use_container_width=True,
            hide_index=True,
        )

    with tab_features:
        if result.feature_importance is None:
            st.warning(t("no_importance", lang))
        else:
            st.subheader(t("feature_importance", lang))
            st.caption(t("top_feature_help", lang))
            importance = result.feature_importance.head(20)
            fig_imp = px.bar(
                importance.sort_values("importance"),
                x="importance",
                y="feature",
                orientation="h",
                template="plotly_dark",
            )
            fig_imp.update_layout(
                height=520,
                paper_bgcolor="#0f172a",
                plot_bgcolor="#111827",
                margin=dict(l=20, r=20, t=20, b=20),
            )
            fig_imp.update_xaxes(gridcolor="#334155", zerolinecolor="#334155")
            fig_imp.update_yaxes(gridcolor="#334155", zerolinecolor="#334155")
            st.plotly_chart(fig_imp, use_container_width=True, config=chart_config)
            st.dataframe(build_feature_table(importance, lang), use_container_width=True, hide_index=True)

    with tab_explain:
        sample_features = result.feature_importance.head(20) if result.feature_importance is not None else pd.DataFrame(
            {"feature": get_feature_columns(dataset)[:20], "importance": [0] * min(20, len(get_feature_columns(dataset)))}
        )
        st.dataframe(build_feature_table(sample_features, lang), use_container_width=True, hide_index=True)


def main() -> None:
    apply_theme()

    with st.sidebar:
        language_choice = st.selectbox("ภาษา / Language", ["ไทย", "English"], index=0)
        lang = "TH" if language_choice == "ไทย" else "EN"

        st.divider()
        st.subheader(t("control_panel", lang))

        with st.form("analysis_controls"):
            quick_ticker = st.selectbox(
                t("quick_pick", lang),
                POPULAR_TICKERS,
                index=POPULAR_TICKERS.index(DEFAULT_TICKER),
            )
            use_custom = st.checkbox(t("custom_ticker", lang), value=False)
            typed_ticker = st.text_input(t("ticker", lang), value=DEFAULT_TICKER)
            range_label_to_key = {RANGE_LABELS[lang][key]: key for key in RANGE_KEYS}
            selected_range_label = st.selectbox(
                t("range", lang),
                list(range_label_to_key.keys()),
                index=RANGE_KEYS.index("1Y"),
            )
            range_key = range_label_to_key[selected_range_label]

            model_labels = {label: key for key, label in TEXT[lang]["model_names"].items()}
            selected_model_label = st.selectbox(t("model", lang), list(model_labels.keys()), index=1)
            model_name = model_labels[selected_model_label]
            tune = st.toggle(t("tune", lang), value=False)
            submitted = st.form_submit_button(t("run", lang), type="primary", use_container_width=True)

        st.markdown(f"<div class='fine-print'>{t('disclaimer', lang)}</div>", unsafe_allow_html=True)

    ticker = (typed_ticker if use_custom else quick_ticker).strip().upper()
    render_header(ticker, lang)

    if submitted:
        with st.spinner(t("loading", lang)):
            try:
                st.session_state["analysis_result"] = run_analysis(ticker, range_key, model_name, tune)
            except ValueError as exc:
                if str(exc) == "not_enough":
                    st.error(t("not_enough", lang))
                    return
                raise

    state = st.session_state.get("analysis_result")
    if state is None:
        render_empty_state(ticker, selected_model_label, range_key, tune, lang)
        return

    render_results(state, lang)


if __name__ == "__main__":
    main()
