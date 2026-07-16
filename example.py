import pandas as pd
import matplotlib.pyplot as plt

from backtester import Backtester
from strategies import BuyAndHold

plt.style.use('dark_background')

MINUTES_PER_DAY = 60 * 24
MINUTES_PER_MONTH = MINUTES_PER_DAY * 30
MINUTES_PER_YEAR = MINUTES_PER_DAY * 365

# Load csv containing a column of close prices. This one has minute intervals
df = pd.read_csv("data/SOLUSDT-1.csv", sep="|", index_col=0) 

# Select the last month of data for example purposes
df = df.iloc[-MINUTES_PER_MONTH:].reset_index() 
    
# Create Backtester object
backtester = Backtester(
    df=df, close_column="close", # Name of close price column in your data
    bars_per_year=MINUTES_PER_YEAR, 
    windows_per_year=12, # 12 months per year
    fee_rate=0.00001, fee_in_usd=False # False since we specify fees in SOL
)

# Example buy and hold strategy holding 2 SOL
backtester.run(BuyAndHold(amount=2.0), n_mote_carlo_paths=1000)

# Baseline strategy
backtester.run(BuyAndHold(amount=1.0), n_mote_carlo_paths=0, baseline=True)

# Display strategy performance
backtester.plot_results(figsize=(12, 8), plot_correlations=False)
backtester.log_results()