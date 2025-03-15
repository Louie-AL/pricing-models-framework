"""
Pricing models for valuation of assets.
"""


from abc import ABC, abstractmethod
import datetime
import numpy as np
import pandas as pd
import warnings
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class PricingModel(ABC):
    """Base class for all pricing models

    Each concrete pricing model must inherit from this class and implement the price method.
    """

    @abstractmethod
    def price(self, data) -> float:
        pass


# class DiscountedCashFlow(PricingModel):
#     """Discounted cash flow pricing model"""

#     def __init__(
#         self,
#         # pricing_date: datetime.date | int,
#         pricing_date: datetime.date,
#         discount_rate: float,
#         compound_frequency: str = 'c',
#     ):
#         """
#         Args:
#             pricing_date: datetime.date or int.
#                 Date of pricing.
#             discount_rate: float.
#                 Discount rate for the DCF model.
#             compound_frequency: str.
#                 Frequency of compounding. Can take the following values:
#                     - 'c': continuous compounding.
#                     - 'a': annual compounding.
#                     - 'sa': semi-annual compounding.
#                     - 'm': monthly compounding.
#                 Default is 'c'.
#         """
#         self._discount_rate = discount_rate
#         self._pricing_date = pricing_date
#         self._compound_frequency = compound_frequency

#     def price(self, data) -> float:
#         """
#         Price the asset using the DCF model.

#         Args:
#             data: pd.DataFrame.
#                 Data to be used for pricing. Must contain the following columns:
#                 - date: datetime.date or int. Date of the cash flow.
#                 - cash_flow: float. Amount of the cash flow.

#         Returns:
#             float. Price of the asset.

#         Raises:
#             ValueError: If the compound_frequency is not one of 'c', 'a', 'sa', 'm'.
#         """
#         # Calculate the present value of cash flows
#         if isinstance(data['date'].iloc[0], float) or isinstance(data['date'].iloc[0], int):
#             t = data['date']
#         else:
#             data['date'] = pd.to_datetime(data['date'])
#             pricing_date_ts = pd.Timestamp(self._pricing_date)
#             t = (data['date'] - pricing_date_ts).apply(lambda x: x.days) / 365

#         present_value = None
#         if self._compound_frequency == 'c':
#             present_value = data['cash_flow'] / np.exp(self._discount_rate * t)
#         elif self._compound_frequency == 'a':
#             present_value = data['cash_flow'] / (1 + self._discount_rate) ** t
#         elif self._compound_frequency == 'sa':
#             present_value = data['cash_flow'] / (
#                 1 + self._discount_rate / 2
#             ) ** (2 * t)
#         elif self._compound_frequency == 'm':
#             present_value = data['cash_flow'] / (
#                 1 + self._discount_rate / 12
#             ) ** (12 * t)
#         else:
#             raise ValueError(
#                 f'Invalid compound frequency: {self._compound_frequency}.\n'
#                 + f'Valid values are: c(continuous), a(annually), sa(semi-annually), m(monthly).'
#             )
#         return present_value.sum()
    
class ClosedFormPricingModel(PricingModel):
    """Closed-form pricing model for assets"""

    def __init__(self, 
                 discount_rate: float,
                 LTM: int,
                 age: float,
                 param: tuple
                 ):
        """
        Args:
            discount_rate: float.
                Discount rate for the closed-form pricing model.
            LTM: int.
                Last twelve months revenue.
            age: float.
                Age of the deal.

        """
        self._discount_rate = discount_rate
        self._LTM = LTM
        self._age = age
        self._param = param

    def price(self) -> float:
        """
        Price the asset using the closed-form pricing model.

        Returns:
            float. Price of the asset.
        """
        a = self._param[0]
        b = self._param[1]
        c = self._param[2]

        numerator_term1 = a * np.exp(-b * (self._age - 1)) / (1 + self._discount_rate - np.exp(-b))
        numerator_term2 = c / self._discount_rate
        numerator = numerator_term1 + numerator_term2

        denominator = a * np.exp(-b * (self._age - 2)) + c

        price = (self._LTM / denominator) * numerator
        return price

class DiscountRateCalibrator:
    """A class to calibrate the discount rate for each deal"""
    def __init__(self, df, forecast_model, plot=False):
        self.df = df
        self.forecast_model = forecast_model
        self.age_rate_map = None
        self.inverse_compute(plot=plot)
    
    def get_age_rate_map(self):
        return self.age_rate_map
    
    def inverse_compute(self, plot=True, ):
        """Inverse compute the discount rate for each deal in the dataframe, and plot the discount rate against the age of the deal
    
        Args:
            plot (bool): whether to plot the discount rate against age
        Returns:
            An age_rate_map that maps the age to the discount rate
        Output:
            plot of discount rate against age if plot is True.
        
        Notice that df is modified in place (a discount_rate column is added).
        """
        for idx in tqdm(range(len(self.df)), desc='Calibrating discount rates'):
            entry = self.df.iloc[idx]
            cashflow_table = self.forecast_model.forecast(entry)

            def f(r):
                dcf = DiscountedCashFlow(
                    pricing_date=0,
                    discount_rate=r,
                    compound_frequency='a'
                )
                return dcf.price(cashflow_table) - entry['price']

            warnings.filterwarnings('error')
            try:
                r = fsolve(f, .1)[0]
            except RuntimeWarning:
                r = np.nan
            self.df.loc[idx, 'discount_rate'] = r
        
        self.df.dropna(subset=['discount_rate'], inplace=True)
        # plot the discount rate against the age
        if plot:
            plt.figure(figsize=(10, 6))
            plt.scatter(self.df['age'], self.df['discount_rate'], s=8)
            plt.xlabel('Age')
            plt.ylabel('Discount Rate')
            plt.title('Discount Rate vs Age for Life of Rights')
            plt.show()
        self.age_rate_map = self.df[['age', 'discount_rate']].copy()
        self.age_rate_map.sort_values(by='age', inplace=True)
    
    def rate(self, age, aggregation_method='mean'):
        """Return the discount rate for a given age
        
        Args:
            age (float): the age of the deal
            aggregation_method (str): the method to aggregate the discount rates. Can be 'mean' or 'median'. Default is 'mean'.
        
        Returns:
            float: the discount rate for the given age
        """
        # Check if the age_rate_map is already computed
        if not hasattr(self, 'age_rate_map'):
            raise ValueError('Please run inverse_compute first')

        if aggregation_method == 'mean':
            if age >= 10:
                return self.age_rate_map[self.age_rate_map['age'] >= 10]['discount_rate'].mean()
            if age < 2:
                return self.age_rate_map[self.age_rate_map['age'] < 2]['discount_rate'].mean()
            elif age < 4:
                return self.age_rate_map[(self.age_rate_map['age'] >= 2) & (self.age_rate_map['age'] < 4)]['discount_rate'].mean()
            elif age < 6:
                return self.age_rate_map[(self.age_rate_map['age'] >= 4) & (self.age_rate_map['age'] < 6)]['discount_rate'].mean()
            elif age < 8:
                return self.age_rate_map[(self.age_rate_map['age'] >= 6) & (self.age_rate_map['age'] < 8)]['discount_rate'].mean()
            else:
                return self.age_rate_map[(self.age_rate_map['age'] >= 8) & (self.age_rate_map['age'] < 10)]['discount_rate'].mean() 

        elif aggregation_method == 'median':
            if age >= 10:
                return self.age_rate_map[self.age_rate_map['age'] >= 10]['discount_rate'].median()
            if age < 2:
                return self.age_rate_map[self.age_rate_map['age'] < 2]['discount_rate'].median()
            elif age < 4:
                return self.age_rate_map[(self.age_rate_map['age'] >= 2) & (self.age_rate_map['age'] < 4)]['discount_rate'].median()
            elif age < 6:
                return self.age_rate_map[(self.age_rate_map['age'] >= 4) & (self.age_rate_map['age'] < 6)]['discount_rate'].median()
            elif age < 8:
                return self.age_rate_map[(self.age_rate_map['age'] >= 6) & (self.age_rate_map['age'] < 8)]['discount_rate'].median()
            else:
                return self.age_rate_map[(self.age_rate_map['age'] >= 8) & (self.age_rate_map['age'] < 10)]['discount_rate'].median()
            



class DiscountRateCalibrator2:
    """A class to calibrate the discount rate for each deal"""
    def __init__(self, df, forecast_model, plot=False):
        self.df = df
        self.forecast_model = forecast_model
        self.age_rate_map = None
        self.inverse_compute(plot=plot)
    
    def get_age_rate_map(self):
        return self.age_rate_map
    
    def inverse_compute(self, plot=True, ):
        """Inverse compute the discount rate for each deal in the dataframe, and plot the discount rate against the age of the deal
    
        Args:
            plot (bool): whether to plot the discount rate against age
        Returns:
            An age_rate_map that maps the age to the discount rate
        Output:
            plot of discount rate against age if plot is True.
        
        Notice that df is modified in place (a discount_rate column is added).
        """
        for idx in tqdm(range(len(self.df)), desc='Calibrating discount rates'):
            entry = self.df.iloc[idx]
            cashflow_table = self.forecast_model.forecast(entry)
            a = self.forecast_model.a
            b = self.forecast_model.b
            c = self.forecast_model.c
            def f(r):

                closed_form = ClosedFormPricingModel(
                    discount_rate=r,
                    LTM=entry['LTM'],
                    age = entry['age'],
                    param = (a, b, c)
                    )
                
                return closed_form.price() - entry['price']

            warnings.filterwarnings('error')
            try:
                r = fsolve(f, .1)[0]
            except RuntimeWarning:
                r = np.nan
            self.df.loc[idx, 'discount_rate'] = r
        
        self.df.dropna(subset=['discount_rate'], inplace=True)
        # plot the discount rate against the age
        if plot:
            plt.figure(figsize=(10, 6))
            plt.scatter(self.df['age'], self.df['discount_rate'], s=8)
            plt.xlabel('Age')
            plt.ylabel('Discount Rate')
            plt.title('Discount Rate vs Age for Life of Rights')
            plt.show()
        self.age_rate_map = self.df[['age', 'discount_rate']].copy()
        self.age_rate_map.sort_values(by='age', inplace=True)
    
    def rate(self, age, aggregation_method='mean'):
        """Return the discount rate for a given age
        
        Args:
            age (float): the age of the deal
            aggregation_method (str): the method to aggregate the discount rates. Can be 'mean' or 'median'. Default is 'mean'.
        
        Returns:
            float: the discount rate for the given age
        """
        # Check if the age_rate_map is already computed
        if not hasattr(self, 'age_rate_map'):
            raise ValueError('Please run inverse_compute first')

        if aggregation_method == 'mean':
            if age >= 10:
                return self.age_rate_map[self.age_rate_map['age'] >= 10]['discount_rate'].mean()
            if age < 2:
                return self.age_rate_map[self.age_rate_map['age'] < 2]['discount_rate'].mean()
            elif age < 4:
                return self.age_rate_map[(self.age_rate_map['age'] >= 2) & (self.age_rate_map['age'] < 4)]['discount_rate'].mean()
            elif age < 6:
                return self.age_rate_map[(self.age_rate_map['age'] >= 4) & (self.age_rate_map['age'] < 6)]['discount_rate'].mean()
            elif age < 8:
                return self.age_rate_map[(self.age_rate_map['age'] >= 6) & (self.age_rate_map['age'] < 8)]['discount_rate'].mean()
            else:
                return self.age_rate_map[(self.age_rate_map['age'] >= 8) & (self.age_rate_map['age'] < 10)]['discount_rate'].mean() 

        elif aggregation_method == 'median':
            if age >= 10:
                return self.age_rate_map[self.age_rate_map['age'] >= 10]['discount_rate'].median()
            if age < 2:
                return self.age_rate_map[self.age_rate_map['age'] < 2]['discount_rate'].median()
            elif age < 4:
                return self.age_rate_map[(self.age_rate_map['age'] >= 2) & (self.age_rate_map['age'] < 4)]['discount_rate'].median()
            elif age < 6:
                return self.age_rate_map[(self.age_rate_map['age'] >= 4) & (self.age_rate_map['age'] < 6)]['discount_rate'].median()
            elif age < 8:
                return self.age_rate_map[(self.age_rate_map['age'] >= 6) & (self.age_rate_map['age'] < 8)]['discount_rate'].median()
            else:
                return self.age_rate_map[(self.age_rate_map['age'] >= 8) & (self.age_rate_map['age'] < 10)]['discount_rate'].median()