from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

class Strategy(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        ...

@dataclass(slots=True)
class BuyAndHold(Strategy):
    amount: float = 1.0

    @property
    def name(self) -> str:
        return f"Buy & Hold {self.amount} Tokens"
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(self.amount, index=range(len(df)))