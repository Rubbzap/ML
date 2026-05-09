# Stock Price Movement Forecasting

โปรเจกต์นี้เป็นระบบพยากรณ์ทิศทางราคาหุ้นขึ้น/ลงด้วย Time Series และ Machine Learning พร้อม dashboard สำหรับทดลองใช้งาน

> หัวข้อ: Stock Price Movement Forecasting Using Hybrid Time Series and Machine Learning Models with Interactive Dashboard Deployment

## เป้าหมาย

- ดึงข้อมูลราคาหุ้นย้อนหลังจาก Yahoo Finance
- สร้าง feature จาก time series และ technical indicators
- เปรียบเทียบโมเดล baseline, Logistic Regression, Random Forest และ XGBoost
- ทำ hyperparameter tuning ด้วย time-series cross validation
- ประเมินผลด้วย classification metrics และ backtesting เบื้องต้น
- Deploy dashboard ด้วย Streamlit
- มี R script สำหรับ EDA และ time series analysis เสริม

## โครงสร้างไฟล์

```text
.
├── README.md
├── project_proposal.md
├── requirements.txt
├── .gitignore
├── data/
│   └── .gitkeep
├── models/
│   └── .gitkeep
├── reports/
│   └── .gitkeep
├── src/
│   ├── app.py
│   ├── config.py
│   ├── data_loader.py
│   ├── features.py
│   ├── modeling.py
│   ├── backtesting.py
│   └── train.py
└── r/
    └── stock_time_series_analysis.R
```

## การติดตั้ง Python

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train โมเดล

```bash
python src/train.py --ticker AAPL --start 2015-01-01 --end 2026-01-01 --model random_forest
```

ตัวอย่างโมเดล:

- `logistic`
- `random_forest`
- `xgboost`

## เปิด Dashboard

```bash
streamlit run src/app.py
```

Dashboard รองรับภาษาไทย/English และมีหน้าจอใช้งานแบบง่ายสำหรับ user ทั่วไป:

- เลือกหุ้นยอดนิยม หรือพิมพ์ ticker เอง
- เลือกช่วงวันที่และโมเดล
- ดูสัญญาณล่าสุด, metrics, candlestick chart, volume, backtest และ feature importance

## การใช้งาน R

ติดตั้ง package ที่จำเป็นใน R:

```r
install.packages(c("tidyverse", "quantmod", "forecast", "tseries", "TTR"))
```

จากนั้นรัน:

```r
source("r/stock_time_series_analysis.R")
```

## หมายเหตุสำคัญ

ผลลัพธ์จากโปรเจกต์นี้ใช้เพื่อการศึกษาเท่านั้น ไม่ใช่คำแนะนำการลงทุนจริง ราคาหุ้นมี noise สูงและได้รับผลกระทบจากเหตุการณ์ภายนอกจำนวนมาก
