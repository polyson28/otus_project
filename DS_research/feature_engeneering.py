import pandas as pd
import numpy as np
from typing import List, Union
from sklearn.preprocessing import StandardScaler

class FeatureEngineer:
    def __init__(self, df: pd.DataFrame, date_col: str = 'date'):
        """
        Инициализация данных, содержащих как минимум следующие столбцы:
        date_col, 'income', 'outcome', 'balance', 
        а также любые существующие макро признаки
        """
        self.df = df.copy()
        self.date_col = date_col
        self.df = self.df.set_index(date_col).asfreq('D')
        self.features = pd.DataFrame(index=self.df.index)
    
    def add_lag_features(self, lags: List[int] = [1, 2, 7]) -> 'FeatureEngineer':
        """
        Добавление признаков лагов для указанных периодов лагов
        """
        for lag in lags:
            self.features[f'balance_lag{lag}'] = self.df['balance'].shift(lag)
            self.features[f'income_lag{lag}'] = self.df['income'].shift(lag)
            self.features[f'outcome_lag{lag}'] = self.df['outcome'].shift(lag)
        return self
    
    def add_rolling_features(self, windows: List[int] = [3, 7, 30]) -> 'FeatureEngineer':
        """
        Добавление скользящего окна для указанных размеров окон (в днях)
        """
        for w in windows:
            # shift(1) для того чтобы текущий элемент не входил в окно
            self.features[f'balance_ma{w}'] = self.df['balance'].shift(1).rolling(window=w, min_periods=1).mean()
            self.features[f'income_ma{w}'] = self.df['income'].shift(1).rolling(window=w, min_periods=1).mean()
            self.features[f'outcome_ma{w}'] = self.df['outcome'].shift(1).rolling(window=w, min_periods=1).mean()
        return self
    
    def add_seasonal_features(self) -> 'FeatureEngineer':
        """
        Добавление сезонных индикаторов, таких как день недели и месяц
        """
        self.features['day_of_week'] = self.features.index.dayofweek
        self.features['month'] = self.features.index.month
        self.features['day_of_week_sin'] = np.sin(2 * np.pi * self.features['day_of_week'] / 7)
        self.features['day_of_week_cos'] = np.cos(2 * np.pi * self.features['day_of_week'] / 7)
        self.features['month_sin'] = np.sin(2 * np.pi * self.features['month'] / 12)
        self.features['month_cos'] = np.cos(2 * np.pi * self.features['month'] / 12)

        self.features = pd.get_dummies(self.features, columns=['day_of_week'], prefix='dow', drop_first=False)
        return self
    
    def add_special_dates(self, tax_dates: Union[set, List[Union[pd.Timestamp, str]]]) -> 'FeatureEngineer':
        """
        Добавление бинарного признака из налогового календаря
        """
        self.features['tax_day'] = 0
        tax_dates = pd.to_datetime(tax_dates)
        self.features.loc[self.features.index.isin(tax_dates), 'tax_day'] = 1
        return self
    
    def add_macro_features(self, macro_df: pd.DataFrame) -> 'FeatureEngineer':
        """
        Добавление макро переменных (уже выровненных по дате)
        """
        macro_df_adj = macro_df.copy()
        numeric_cols = [
            col for col in macro_df_adj if 
            pd.api.types.is_numeric_dtype(macro_df_adj[col]) and
            len(macro_df_adj[col].unique()) > 2 # не стандартизируем бинарные 
        ]
        if numeric_cols:
            scaler = StandardScaler()
            macro_df_adj[numeric_cols] = scaler.fit_transform(macro_df_adj[numeric_cols])
            
        overlap = set(macro_df_adj.columns).intersection(self.features.columns)
        if overlap:
            self.features = self.features.drop(columns=overlap)
            
        self.features = self.features.join(macro_df_adj, how='left')
        # forward-fill для заполнения последнего известного значения, если данные не ежедневные
        self.features.fillna(method='ffill', inplace=True)
        return self
    
    def get_feature_df(self) -> tuple[pd.DataFrame, pd.Series]:
        """
        Возврат конечного датасета признаков 
        """
        # Удаление строк с NaN (из-за лагов) в начале
        feature_df = self.features.copy()
        feature_df.dropna(inplace=True)
        valid_indices = feature_df.index
        balance_series = self.df.loc[valid_indices, 'balance']
        balance_series.dropna(inplace=True)
        valid_balance_indices = balance_series.index
        feature_df = feature_df.loc[valid_balance_indices]

        return feature_df, balance_series