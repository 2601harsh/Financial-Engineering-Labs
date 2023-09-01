import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")

# Read data from CSV files
bse_m = pd.read_csv('./BSE/M_stocks_in_sensex.csv', usecols=range(1, 11))
bse_w = pd.read_csv('./BSE/W_stocks_in_sensex.csv', usecols=range(1, 11))
bse_d = pd.read_csv('./BSE/stocks_in_sensex.csv', usecols=range(1, 11))
nse_m = pd.read_csv('./NSE/M_stocks_in_nifty.csv', usecols=range(1, 11))
nse_w = pd.read_csv('./NSE/W_stocks_in_nifty.csv', usecols=range(1, 11))
nse_d = pd.read_csv('./NSE/stocks_in_nifty.csv', usecols=range(1, 11))

bse_data = [bse_m, bse_w, bse_d]
nse_data = [nse_m, nse_w, nse_d]
time_intervals = ['Monthly', 'Weekly', 'Daily']
colors = ['orange', 'blue', 'green']
market_names = ['BSE', 'NSE']

# Plot the data for each market and time interval
for market_idx, market_data in enumerate([bse_data, nse_data]):
    for stock_idx in range(10):
        for time_idx, time_data in enumerate(market_data):
            time_points = len(time_data)
            stock_name = time_data.columns[stock_idx]
            plt.plot(range(time_points), time_data.iloc[:, stock_idx], color=colors[time_idx])
            plt.xlabel(f'Time Points ({time_intervals[time_idx]} basis)')
            plt.ylabel('Stock Prices')
            plt.title(f'Stock Prices vs Time ({time_intervals[time_idx]}) for {market_names[market_idx]} - {stock_name}')
            plt.show()
