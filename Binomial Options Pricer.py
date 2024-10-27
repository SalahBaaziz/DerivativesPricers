import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

def binomial_option_price(spot, K, T, r, sigma, n, option_type="call"):
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    asset_prices = np.zeros(n + 1)
    for j in range(n + 1):
        asset_prices[j] = spot * (u ** (n - j)) * (d ** j)

    option_values = np.zeros(n + 1)
    if option_type == "call":
        option_values = np.maximum(0, asset_prices - K)
    elif option_type == "put":
        option_values = np.maximum(0, K - asset_prices)

    for i in range(n - 1, -1, -1):
        option_values = np.exp(-r * dt) * (p * option_values[:-1] + (1 - p) * option_values[1:])

    return option_values[0]

def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    spot = stock.history(period='1d')['Close'].iloc[-1]
    r = 0.045
    return spot, r

def get_iv(ticker, K, T):
    stock = yf.Ticker(ticker)
    options = stock.option_chain()
    calls = options.calls
    puts = options.puts

    iv = None
    for index, row in calls.iterrows():
        if row['strike'] == K:
            iv = row['impliedVolatility']
            break

    if iv is None:
        for index, row in puts.iterrows():
            if row['strike'] == K:
                iv = row['impliedVolatility']
                break

    return iv

def calculate_option_prices(tickers, T, percent_change, n):
    results = []
    for ticker in tickers:
        try:
            S0, r = get_stock_data(ticker)
            K = round(S0 * (1 + percent_change / 100) / 5) * 5  # match K with yfinance's strike prices for data extrcting
            sigma = get_iv(ticker, K, T)

            if sigma is None:
                print(f"Implied volatility not found for {ticker} at strike {K}.")
                continue

            call_price = binomial_option_price(S0, K, T, r, sigma, n, option_type="call")
            put_price = binomial_option_price(S0, K, T, r, sigma, n, option_type="put")
            straddle_price = call_price + put_price
            ratio = call_price / put_price if put_price != 0 else np.nan

            results.append({
                "Stock": ticker,
                "Current Price": round(S0),
                "Strike Price": K,
                "Implied Volatility": sigma,
                "Call Price": call_price,
                "Put Price": put_price,
                "Straddle Price": straddle_price,
                "Ratio": ratio
            })
        except Exception as e:
            print(f"Error retrieving data for {ticker}: {e}")

    return results

tickers_input = input("Enter stock tickers separated by commas (e.g., AAPL, MSFT, TSLA): ")
tickers = [ticker.strip().upper() for ticker in tickers_input.split(",")]
T = float(input("Enter the time to expiration in days (T): ")) / 365
percent_change = float(input("Enter the percentage change for the strike price (e.g., 5 for +5%): "))
n = int(input("Enter the number of steps for the binomial model: "))

results = calculate_option_prices(tickers, T, percent_change, n)
results_df = pd.DataFrame(results)

print("\nOptions Pricing Table:")
print(results_df)
