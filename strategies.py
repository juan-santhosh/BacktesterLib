from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

class Strategy(ABC):
    """
    Abstract class which respresents a vectorised trading strategy to be evaluated with the backtester.
    Consists of a name and a signal function to return target positions over time given price data.
    """
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Vectorised signal functions which takes a historical price DataFrame and returns a signal Series.

        Args:
            df (pd.DataFrame): DataFrame containing historical close price data.
        
        Returns:
            pd.Series: Series of target positions over time corresponding to each index in the DataFrame.
        """
        ...

@dataclass(slots=True)
class BuyAndHold(Strategy): # Example strategy
    amount: float = 1.0

    @property
    def name(self) -> str:
        return f"Buy & Hold {self.amount} Tokens"
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        # Hold {amount} tokens over entire window.
        return pd.Series(self.amount, index=range(len(df))) 