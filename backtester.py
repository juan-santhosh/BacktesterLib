from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

from strategies import Strategy
from result import BacktestResult

@dataclass(slots=True)
class Backtester:
    df: pd.DataFrame
    close_column: str

    bars_per_year: float
    windows_per_year: float

    fee_rate: float
    fee_in_usd: bool = True

    results: list[BacktestResult] = field(default_factory=list)
    baseline: BacktestResult | None = None

    def run(
            self, strategy: Strategy, baseline: bool = False,
            n_mote_carlo_paths: int = 1000) -> None:
        
        result_df = self.df.copy()

        result_df["market_return"] = (
            result_df[self.close_column] 
            / result_df[self.close_column].shift(1)
        )

        result_df["market_return"] = result_df["market_return"].fillna(1.0)
        result_df["signal"] = strategy.generate_signals(result_df)
        result_df["position"] = result_df["signal"].shift(1).fillna(0.0)

        result_df["trade"] = (
            result_df["position"].diff().fillna(result_df["position"])
        )

        result_df["fee"] = result_df["trade"].abs() * self.fee_rate

        if self.fee_in_usd:
            result_df["fee"] /= result_df[self.close_column]

        result_df["price_log_return"] = pd.Series(
            np.log(result_df[self.close_column])
        ).diff().fillna(0.0)

        result_df["log_return"] = (
            result_df["position"] * result_df["price_log_return"] 
            - result_df["fee"]
        )

        result_df["cum_log_return"] = result_df["log_return"].cumsum()

        result_df["return"] = np.exp(result_df["log_return"])
        result_df["cum_return"] = np.exp(result_df["cum_log_return"])

        if n_mote_carlo_paths > 0:
            log_returns = result_df["log_return"].to_numpy()
            path_length = len(log_returns)

            rng = np.random.default_rng(42)

            indices = rng.integers(
                0, path_length, size=(n_mote_carlo_paths, path_length)
            )

            sample_paths = np.cumsum(log_returns[indices], axis=1)
            t = np.arange(sample_paths.shape[1])

            result_df["mean_path"] = np.exp(np.mean(sample_paths, axis=0))

        else:
            result_df["mean_path"] = None
        
        result = BacktestResult(
            strategy=strategy, result=result_df, 
            bars_per_year=self.bars_per_year,
            windows_per_year=self.windows_per_year
        )

        if not baseline:
            self.results.append(result)
        else:
            self.baseline = result
    
    def plot_results(
            self, figsize: tuple[int, int], 
            plot_correlations: bool = True) -> None:
        
        rows = 1 + len(self.results) if plot_correlations else 1
        fig, axes = plt.subplots(rows, 1, figsize=figsize)

        ax1 = axes[0] if rows > 1 else axes

        ax1.set_ylabel("Cumulative Return")
        ax1.set_xlabel("Timestep")

        ax1.xaxis.set_minor_locator(plticker.AutoMinorLocator(10))
        ax1.yaxis.set_minor_locator(plticker.AutoMinorLocator(10))

        ax1.grid(
            visible=True, which='minor', 
            linestyle=':', linewidth=0.5, alpha=0.2
        )

        ax1.grid(
            visible=True, which='major', 
            linestyle='-', linewidth=1.0, alpha=0.2
        )

        if self.baseline is not None:
            self.baseline.plot_performance(
                ax1, colour="limegreen", alpha=0.8, plot_mean=False
            )

        for result in self.results:
            result.plot_performance(ax=ax1, colour="gold")

        if rows > 1:
            for i, ax in enumerate(axes.flat):
                if i == 0:
                    continue

                ax.set_ylabel("Strategy Return")
                ax.set_xlabel("Market Return")

                result = self.results[i - 1]
                result.plot_return_corr(ax)

        fig.suptitle("Equity Curve")

        plt.tight_layout()
        plt.show()

    def log_results(self) -> None:
        for result in self.results:
            result.log_metrics()

        if self.baseline is not None:
            self.baseline.log_metrics()