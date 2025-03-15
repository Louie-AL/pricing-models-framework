"""
Forecast models for time series data.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import numpy.typing as npt


class ForecastModel(ABC):
    """Base class for all forecast models

    Each concret forecast model must inherit from this class and implement the forecast method.
    """

    @abstractmethod
    def forecast(self, data: npt.ArrayLike) -> pd.DataFrame:  # type: ignore
        pass

class ExponentialDecayForecast(ForecastModel):
    """Exponential decay forecast model

    This model forecasts the next value in a time series by taking a weighted average of all previous values.
    The weights are exponentially decreasing and are defined by the decay factor alpha.
    """

    def __init__(self, a, b, c) -> None:
        self.a = a
        self.b = b
        self.c = c
        self.decay_rate_func = lambda x: a * np.exp(-b * (x - 2)) + c

    def forecast(self, data: pd.Series) -> pd.DataFrame:  # type: ignore
        """Forecast the next value in the time series

        Args:
            data: a pandas Series. The series must contain the LTM, age, and years_remaining.

        Returns:
            A pandas DataFrame that stores the cashflow table.
        """
        cashflow_table = pd.DataFrame(columns=['date', 'cash_flow'])
        try:
            years_remaining = int(data['years_remaining'])
        except ValueError:
            raise ValueError(
                f'Invalid value for years_remaining: {data["years_remaining"]}. Must be an integer.\n' + 
                f'This data point is {data}.'
            )
        for i in range(1, int(data['years_remaining']) + 1):  
            decay_rate = self.decay_rate_func(data['age'] + i) / self.decay_rate_func(data['age'])
            cashflow_table.loc[i] = [i, data['LTM'] * (decay_rate)]
        return cashflow_table
