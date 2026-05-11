import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.tsa.arima.model import ARIMA

class FallbackModel:
    def __init__(self, model_cls=GradientBoostingRegressor, data_path='../Data/chosen_features.xlsx', target_col="balance", confirm=True, **model_kwargs):
        """
        Инициализирует объект для обучения пользовательской модели.
        """
        self.model_cls = model_cls
        self.data_path = data_path
        self.target_col = target_col
        self.confirm = confirm
        self.model_kwargs = model_kwargs
        self.model = None

    def train(self):
        """
        Обучает пользовательскую модель при переходе на ручное управление.
        """
        df = pd.read_excel(self.data_path)
        df.drop('date', inplace=True, axis=1)
        y = df[self.target_col]
        x = df.drop(columns=self.target_col)
        self.model = self.model_cls(**self.model_kwargs)

        print('\n ВНИМАНИЕ!  Вы собираетесь обучить пользовательскую модель:')
        print(f'Класс модели: {self.model.__class__.__name__}')
        print(f'x.shape = {x.shape}, y.shape = {y.shape}')
        print('-------------------------------------------------------------')

        if self.confirm:
            proceed = input("Нажмите <Enter>, чтобы начать обучение, или 'q' для отмены: ")
            if proceed.lower().startswith('q'):
                raise RuntimeError('Обучение пользовательской модели отменено оператором')

        self.model.fit(x, y)
        preds = self.model.predict(x)
        mae = mean_absolute_error(y, preds)
        print(f'✅ Пользовательская модель обучена. MAE (train) = {mae:.4f}')
        return self.model
        

