# Project Proposal

## ชื่อโปรเจกต์

**Stock Price Movement Forecasting Using Hybrid Time Series and Machine Learning Models with Interactive Dashboard Deployment**

## ที่มาและความสำคัญ

ตลาดหุ้นเป็นข้อมูลประเภท time series ที่มีความผันผวนสูงและมี noise จำนวนมาก การทำนายราคาหุ้นแบบจุดต่อจุดมักทำได้ยากและอาจไม่เสถียร ดังนั้นโปรเจกต์นี้จึงเน้นการพยากรณ์ **ทิศทางการเคลื่อนไหวของราคา** ว่าวันถัดไปมีแนวโน้มขึ้นหรือลง โดยใช้ข้อมูลราคา ปริมาณซื้อขาย และ technical indicators ร่วมกับโมเดล Machine Learning

โปรเจกต์นี้เหมาะกับนักศึกษาสถิติและการวิเคราะห์ข้อมูล เพราะครอบคลุมทั้งการวิเคราะห์เชิงสถิติ การสร้าง feature จากข้อมูลอนุกรมเวลา การเปรียบเทียบโมเดล การ tuning และการนำโมเดลไปใช้งานผ่าน dashboard

## วัตถุประสงค์

1. ศึกษารูปแบบการเคลื่อนไหวของราคาหุ้นด้วย exploratory data analysis
2. สร้าง feature จากข้อมูล time series และ technical indicators
3. พัฒนาโมเดลเพื่อทำนายทิศทางราคาหุ้นขึ้น/ลงในวันถัดไป
4. เปรียบเทียบประสิทธิภาพของโมเดลหลายประเภท
5. ทำ hyperparameter tuning ด้วยวิธีที่เหมาะกับ time series
6. ประเมินผลด้วย classification metrics และ backtesting
7. สร้าง dashboard สำหรับเลือกหุ้น ฝึกโมเดล และแสดงผลการพยากรณ์

## ขอบเขตของงาน

- ใช้ข้อมูลหุ้นจาก Yahoo Finance ผ่าน `yfinance`
- รองรับหุ้นต่างประเทศ เช่น AAPL, MSFT, TSLA, NVDA และ ETF เช่น SPY
- เป้าหมายหลักคือการทำนายทิศทางวันถัดไป ไม่ใช่การทำนายราคาปิดแบบ absolute price
- ใช้ Python เป็นแกนหลักสำหรับ machine learning และ deployment
- ใช้ R สำหรับ EDA, visualization และ time series diagnostics เสริม

## ตัวแปรเป้าหมาย

กำหนด target เป็น:

```text
target = 1 ถ้า Close(t+1) > Close(t)
target = 0 ถ้า Close(t+1) <= Close(t)
```

## Features

- Daily return
- Log return
- Lag returns เช่น 1, 2, 3, 5, 10 วัน
- Rolling mean และ rolling volatility
- Moving Average 5, 10, 20, 50 วัน
- RSI
- MACD
- Bollinger Bands
- Volume change

## โมเดลที่ใช้

### Baseline

- Naive direction model
- Majority class model

### Machine Learning

- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

### ส่วนขยายในอนาคต

- LSTM / GRU
- Temporal Convolutional Network
- Transformer-based forecasting
- GARCH สำหรับ volatility forecasting

## Metrics

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Directional Accuracy
- Cumulative return จาก strategy จำลอง
- Sharpe ratio
- Maximum drawdown

## Deployment

ใช้ Streamlit dashboard โดยมีฟีเจอร์:

- เลือก symbol หุ้น
- เลือกช่วงวันที่
- เลือกโมเดล
- แสดงกราฟราคาหุ้น
- แสดงผล metrics
- แสดง confusion matrix
- แสดง feature importance
- แสดง backtesting equity curve
- แสดงสัญญาณล่าสุด: Buy / Hold / Sell แบบง่าย

## ข้อจำกัด

- ข้อมูลราคาหุ้นมีความผันผวนและ noise สูง
- Technical indicators เป็นข้อมูลจากอดีต ไม่สามารถรับประกันอนาคตได้
- ยังไม่รวมข่าว งบการเงิน sentiment หรือ macroeconomic variables
- ผลลัพธ์ใช้เพื่อการศึกษา ไม่ใช่คำแนะนำการลงทุน

## Expected Output

- Source code สำหรับ training และ dashboard
- รายงานผลการวิเคราะห์
- ตารางเปรียบเทียบโมเดล
- Interactive dashboard
- ข้อเสนอแนะสำหรับการพัฒนาต่อ
