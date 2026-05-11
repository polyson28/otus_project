from flaml import AutoML
import pandas as pd
from typing import Any
import joblib


class ModelSelector:
    def __init__(self, time_budget: int = 600, metric: str = 'mae'):

        """
        Класс, который подбирает модельку исходя из передаваемых нами параметров. Используется раз в период для переобучения
        модели или при обнаружении concept drift. 
        time_budget: сколько времени выделяем на перебор параметров и моделей (в секундах)
        metric: метрика, которую хотим оптимизировать
        """
        self.time_budget = time_budget
        self.metric = metric
        self.automl = AutoML()
        self.best_model = None
        self.best_model_name = None
        
    def find_best_model(self, x_train: pd.DataFrame, y_train: pd.Series, period: int = 1) -> Any:
        """
        Запуск AutoML для нахождения лучшей модели. 
        Период прогноза задаем самостоятельно 
        x_train: обучающая выборка 
        y_train: целевая переменная
        period: горизонт прогноза
        """
        self.automl.fit(
            X_train=x_train, y_train=y_train,
            task='ts_forecast',
            metric=self.metric,
            eval_method='holdout',
            time_budget=self.time_budget,
            verbose=1,
            period=period
        )
        self.best_model = self.automl.model.estimator
        self.best_model_name = self.automl.best_estimator
        print(f'AutoML selected model: {self.best_model_name}')
        return self.automl
    
    def save(self, path: str = 'model.pkl'):
        joblib.dump(self.automl, path)
        print(f'Модель сохранена в {path}')