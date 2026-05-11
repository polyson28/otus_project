import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def normal_likelihood(value, mean_0, mean_1, std):
    """
    Логарифм отношения правдоподобий нормальных плотностей
    """
    return np.log(norm.pdf(value, mean_0, std) / norm.pdf(value, mean_1, std))

class ChangeFinder:
    """
    Детектор разладок двумя методами:
    SR (Shiryaev–Roberts) - method='sr'
    CUSUM (Cumulative Sum) - method='cusum'
    """
    def __init__(self,
                 alpha: float,
                 beta: float,
                 method: str = 'sr',
                 sigma_diff: float = None,     # для SR
                 mean_diff: float = None,      # для CUSUM
                 ceil: float = 1e6,            # для SR
                 trsh: float = 2,
                 breaks_max: int = 5,
                 slice_length: int = 15):
        assert method in ('sr', 'cusum'), "method must be 'sr' or 'cusum'"
        self.method = method

        # параметры для экспоненциального сглаживания
        self.alpha = alpha
        self.beta = beta

        # параметры конкретных методов
        self.sigma_diff = sigma_diff
        self.mean_diff = mean_diff
        self.ceil = ceil
        self.trsh = trsh

        # для отслеживания состояний
        self.metric = 0.
        self.states = []
        self.breakpoints = []
        self.slice_length = slice_length
        self.breaks_max = breaks_max

        # текущее оценённое среднее и дисперсия
        self.mean_hat = 0.
        self.var_hat = 1.

    def get_values(self):
        """Вычислить текущие оценки mean_hat и var_hat."""
        try:
            self.mean_hat = self.mean_values_sum / self.mean_weights_sum
            self.var_hat = self.var_values_sum / self.var_weights_sum
        except AttributeError:
            # первый вызов
            self.mean_hat = 0.
            self.var_hat = 1.

    def update(self, new_value: float):
        """
        Принимает новый сэмпл, обновляет оценку среднего и дисперсии,
        а также сохраняет вспомогательные величины для подсчёта метрики.
        """
        self.get_values()

        if self.method == 'sr':
            # для SR: предсказываем квадратичную ошибку
            self.pred_diff_val = (new_value - self.mean_hat)**2
            self.pred_diff_mean = self.var_hat
            new_var_value = self.pred_diff_val
        else:
            # для CUSUM: стандартизируем
            std_hat = np.sqrt(self.var_hat)
            self.new_value_norm = (new_value - self.mean_hat) / std_hat
            new_var_value = (self.new_value_norm - self.mean_hat)**2

        # обновляем сумму и веса для среднего
        try:
            self.mean_values_sum = (1 - self.alpha) * self.mean_values_sum + new_value
            self.mean_weights_sum = (1 - self.alpha) * self.mean_weights_sum + 1.0
        except AttributeError:
            self.mean_values_sum = new_value
            self.mean_weights_sum = 1.0

        # обновляем сумму и веса для дисперсии
        try:
            self.var_values_sum = (1 - self.beta) * self.var_values_sum + new_var_value
            self.var_weights_sum = (1 - self.beta) * self.var_weights_sum + 1.0
        except AttributeError:
            self.var_values_sum = new_var_value
            self.var_weights_sum = 1.0

    def count_metric(self):
        """Вычислить и накопить детекционную метрику, сохранить state и breakpoint."""
        if self.method == 'sr':
            # Shiryaev–Roberts
            adjusted = self.pred_diff_val - self.pred_diff_mean
            likelihood = np.exp(self.sigma_diff * (adjusted - self.sigma_diff / 2.0))
            self.metric = min(self.ceil, (1.0 + self.metric) * likelihood)
        else:
            # CUSUM
            zeta = normal_likelihood(self.new_value_norm, self.mean_diff, 0., 1.)
            self.metric = max(0.0, self.metric + zeta)

        # определить state по порогу
        state = 1 if self.metric > self.trsh else 0
        self.states.append(state)

        # определить breakpoint (цвет) по числу recent 1 в states
        recent = np.array(self.states[-self.slice_length:])
        color = 'red' if recent.sum() > self.breaks_max else 'blue'
        self.breakpoints.append(color)

    def feed(self, new_value: float):
        """
        Удобный метод-наборщик: один шаг update + count_metric
        """
        self.update(new_value)
        self.count_metric()

    @staticmethod
    def plot(data, model, method):
        stat_trajectory = []
        mean_values    = []
        for k, x_k in enumerate(data['balance'].values):
            model.feed(x_k)
            stat_trajectory.append(model.metric)
            mean_values.append(model.mean_hat)

        fig, ax = plt.subplots(figsize=(20, 8))
        
        for i in range(1, len(data['balance'].values)):
            x = [i-1, i]
            y = [data['balance'].values[i-1], data['balance'].values[i]]
            ax.plot(x, y, color=model.breakpoints[i])
    
        legend_elements = [
            Line2D([0], [0], color='blue',  lw=2, label='Норма'),
            Line2D([0], [0], color='red',   lw=2, label='Разладка'),
            Line2D([0], [0], color='black', lw=2, label='Среднее'),
        ]
        ax.plot([i for i in range(1, len(stat_trajectory))], mean_values[1:], color='black',label='Среднее')  

        ax.legend(handles=legend_elements, loc='best')
        plt.title(f'Разладка на исторических значениях сальдо, метод {method}')
        plt.show()

    @staticmethod
    def detect_anomaly_real_time(detector, new_value):
        """
        брабатывает новое значение через детектор и возвращает True, 
        если зафиксирована аномалия (state == 1), иначе False
        """
        detector.feed(new_value)
        return detector.states[-1] == 1