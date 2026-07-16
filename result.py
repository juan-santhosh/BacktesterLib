from dataclasses import dataclass

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from strategies import Strategy

@dataclass(slots=True)
class BacktestResult:

    """Dataclass which stores backtester results.

    Attributes:
        strategy (Strategy): Vectorised trading strategy.
        result (pd.DataFrame): Modified price DataFrame which includes trading activity and performance.
        bars_per_year (float): Number of bars in a year in your data interval.
        windows_per_year (float): Number of backtest windows in a complete year.
    """

    strategy: Strategy
    result: pd.DataFrame
    bars_per_year: float
    windows_per_year: float

    @property
    def equity_curve(self) -> np.ndarray:
        return np.exp(self.result["cum_log_return"].to_numpy())
    
    @property
    def total_return(self) -> float:
        return self.equity_curve[-1]
    
    @property
    def annualised_return(self) -> float:
        return (self.total_return) ** self.windows_per_year

    @property
    def sharpe_ratio(self) -> float:
        returns = self.result["log_return"]

        if returns.std() == 0:
            return 0.0

        return (
            returns.mean()
            / returns.std()
            * np.sqrt(self.bars_per_year)
        )
    
    @property
    def max_drawdown(self) -> float:
        equity = self.equity_curve
        running_max = np.maximum.accumulate(equity)
        drawdown = equity / running_max - 1
        return drawdown.min()
    
    @property
    def drawdown(self) -> np.ndarray:
        equity = self.equity_curve
        running_max = np.maximum.accumulate(equity)
        return equity / running_max - 1
    
    @property
    def win_rate(self) -> float:
        r = self.result["log_return"]
        r = r[r != 0.0]

        return (r > 0).mean()

    def log_metrics(self) -> None:
        print(
            f"\nSTRATEGY: {self.strategy.name}\n"
            f"\nReturn: {self.total_return:.4f}"
            f"\nAnnualised Return: {self.annualised_return:.4f}"
            f"\nSharpe: {self.sharpe_ratio:.4f}"
            f"\nMax Drawdown: {self.max_drawdown:.4f}"
            f"\nWin Rate: {self.win_rate:.4f}\n"
        )

    def plot_performance(
            self, ax: Axes, colour: str, alpha: float = 1.0,
            plot_mean: bool = True, plot_position: bool = False) -> None:
        
        """Plots strategy perfomance alongside optional mean and position over time.

        Args:
            ax (Axes): Matplotlib axes to plot return path on.
            colour (str): Colour of line to plot.
            alpha (float): Opacity of line. 1.0 is opaque while 0.0 is transparent.
            plot_mean (bool): Default True. Decides whether to plot mean return path from monte carlo simulations.
            plot_position (bool): Default False. Decides whether to plot position over time.
        """
        
        sns.lineplot(
            data=self.result, x=self.result.index, y="cum_return",
            ax=ax, color=colour, label=self.strategy.name, alpha=alpha,
        )

        if plot_mean:
            sns.lineplot(
                data=self.result, x=self.result.index, y="mean_path",
                ax=ax, color=colour, alpha=0.5 * alpha, linestyle="--"
            )

        if plot_position:
            sns.lineplot(
                data=self.result, x=self.result.index, y="position",
                ax=ax, color=colour, alpha=0.2 * alpha, linestyle=":"
            )

    def plot_return_corr(self, ax: Axes) -> None:
        """Plots strategy return correlation to market returns.

        Args:
            ax (Axes): Matplotlib axes to plot regression on.
        """
         
        ax.set_xlabel("Market Return")
        ax.set_ylabel("Strategy Return")

        sns.regplot(
            data=self.result, ax=ax, truncate=True,
            x="market_return", y="return"
        )