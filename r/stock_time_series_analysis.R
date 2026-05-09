library(tidyverse)
library(quantmod)
library(forecast)
library(tseries)
library(TTR)

analyze_stock <- function(ticker = "AAPL", start_date = "2015-01-01") {
  getSymbols(ticker, src = "yahoo", from = start_date, auto.assign = TRUE)
  stock_xts <- get(ticker)

  stock_df <- data.frame(
    Date = index(stock_xts),
    coredata(stock_xts)
  )

  colnames(stock_df) <- c("Date", "Open", "High", "Low", "Close", "Volume", "Adjusted")

  stock_df <- stock_df %>%
    mutate(
      return = Adjusted / lag(Adjusted) - 1,
      log_return = log(Adjusted / lag(Adjusted)),
      ma_20 = SMA(Adjusted, n = 20),
      ma_50 = SMA(Adjusted, n = 50),
      rsi_14 = RSI(Adjusted, n = 14)
    ) %>%
    drop_na()

  price_plot <- ggplot(stock_df, aes(Date, Adjusted)) +
    geom_line(color = "#2563eb") +
    geom_line(aes(y = ma_20), color = "#f97316", linewidth = 0.6) +
    geom_line(aes(y = ma_50), color = "#16a34a", linewidth = 0.6) +
    labs(
      title = paste(ticker, "Adjusted Close with Moving Averages"),
      x = "Date",
      y = "Adjusted Close"
    ) +
    theme_minimal()

  return_plot <- ggplot(stock_df, aes(log_return)) +
    geom_histogram(bins = 60, fill = "#475569", color = "white") +
    labs(
      title = paste(ticker, "Log Return Distribution"),
      x = "Log Return",
      y = "Frequency"
    ) +
    theme_minimal()

  print(price_plot)
  print(return_plot)

  cat("\nADF Test for adjusted close:\n")
  print(adf.test(stock_df$Adjusted))

  cat("\nADF Test for log returns:\n")
  print(adf.test(stock_df$log_return))

  cat("\nARIMA model for adjusted close:\n")
  arima_model <- auto.arima(stock_df$Adjusted)
  print(summary(arima_model))

  forecast_result <- forecast(arima_model, h = 30)
  print(autoplot(forecast_result) + theme_minimal())

  invisible(
    list(
      data = stock_df,
      arima_model = arima_model,
      forecast = forecast_result
    )
  )
}

result <- analyze_stock("AAPL", "2015-01-01")
