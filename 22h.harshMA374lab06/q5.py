import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
sns.set(style="darkgrid")

bse_m = pd.read_csv('./BSE/M_stocks_in_sensex.csv', usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
bse_w = pd.read_csv('./BSE/W_stocks_in_sensex.csv', usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
bse_d = pd.read_csv('./BSE/stocks_in_sensex.csv', usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
nse_m = pd.read_csv('./NSE/M_stocks_in_nifty.csv', usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
nse_w = pd.read_csv('./NSE/W_stocks_in_nifty.csv', usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
nse_d = pd.read_csv('./NSE/stocks_in_nifty.csv', usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
bse_m = bse_m.dropna()
bse_w = bse_w.dropna()
bse_d = bse_d.dropna()
nse_m = nse_m.dropna()
nse_w = nse_w.dropna()
nse_d = nse_d.dropna()

(a, c, m, x) = (2057, 1345, pow(2, 50), 3245)

def uniform():
    global x
    x = (a * x + c) % m
    return x / m

def BoxMuller(μ, sigma):
    R = np.sqrt(-2 * np.log(uniform()))
    θ = 2 * np.pi * uniform()
    return μ + sigma * R * np.sin(θ)

def stock_stat_calculator(data):
    length = len(data)
    returns = np.zeros(length - 1)
    for k in range(length - 1):
        returns[k] = np.log(data.iloc[k + 1] / data.iloc[k])
    return np.mean(returns), np.std(returns)

BSE = [bse_m, bse_w, bse_d]
NSE = [nse_m, nse_w, nse_d]
Time_len = [len(BSE[0].iloc[:, 0]), len(BSE[1].iloc[:, 0]), len(BSE[2].iloc[:, 0])]
intervals = ['Monthly', 'Weekly', 'Daily']
colors = ['orange', 'blue', 'green']
Market = [BSE, NSE]
Market_name = ['BSE', 'NSE']

for z in range(2):
    for j in range(2):
        length = len(Market[z][j].iloc[:, 0])
        train_end = int(3 * length / 4)
        data_train = Market[z][j].iloc[0:train_end, 0:10]
        data_test = Market[z][j].iloc[train_end:length, 0:10]
        for i in range(10):
            given_data = data_train.iloc[:, i]
            prediction_data = data_test.iloc[:, i]
            mu, sig = stock_stat_calculator(given_data)
            N = len(prediction_data)
            W = np.zeros(N)
            S = np.zeros(N)
            S[0] = given_data.iloc[-1]
            for k in range(1, N):
                W[k] = W[k - 1] + BoxMuller(0, 1)
                t = k
                S[k] = S[0] * np.exp((mu - (sig**2) / 2) * t + sig * W[k])
            plt.plot(range(train_end - 1, length - 1), S, label='Predicted Stock Prices')
            plt.plot(range(length), Market[z][j].iloc[:, i], label='Actual Stock Price', color=colors[j])
            plt.xlabel(f'Time points ({intervals[j]} basis)')
            plt.ylabel('Stock Prices')
            plt.title(f'Actual vs Predicted Stock Prices ({intervals[j]}) for {Market_name[z]} - {Market[z][j].columns[i]}')
            plt.legend()
            path = f'Plots/Question_5/{Market_name[z]}/{Market[z][j].columns[i]}/'
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(path + f'{intervals[j]}.png')
            plt.show()
