import numpy as np
from typing import Union


def liquidity_decision(
    trues: Union[float, np.ndarray],
    preds: Union[float, np.ndarray],
    key_rate: float = 0.21,
    return_all_values: bool = False
) -> Union[np.ndarray, float]:
    """Функция для описания бизнес логики проекта. 
        Функция рассчитывает прибыль от операций по прогнозированным и реальным сальдо. 

    Args:
        trues (Union[float, np.ndarray]): реальное сальдо на начало дня 
        preds (Union[float, np.ndarray]): предсказанные сальдо 
        key_rate (float, optional): ключевая ставка. Defaults to 0.21.
        return_all_values (bool, optional): если True, то возвращает прибыли по дням, 
            если False, то возвращает сумму. Defaults to False.

    Returns:
        Union[np.ndarray, float]: прибыль по дням (return_all_values = True) 
            или сумму прибыли (return_all_values = False)
    """
    # Ставки
    day_deposit_rate = key_rate + 0.005   # key + 0.5%
    night_deposit_rate = key_rate - 0.009 # key - 0.9%
    night_loan_rate = key_rate + 0.01     # key + 1%

    trues = np.array(trues, copy=True, dtype=float)
    preds = np.array(preds, copy=True, dtype=float)
    profit = np.zeros_like(trues)

    # день
    positive_pred = preds > 0
    # вкладываем прогнозируемый профицит
    profit[positive_pred] += day_deposit_rate * preds[positive_pred]
    trues[positive_pred] -= preds[positive_pred]

    # прогнозируемый дефицит покрываем займом
    trues[~positive_pred] -= preds[~positive_pred]

    # ночь 
    positive_balance = trues > 0
    # если по факту профицит, то депозит в ЦБ overnight
    profit[positive_balance] += night_deposit_rate * trues[positive_balance]
    # если по факту дефицит, заём overnight
    profit[~positive_balance] += night_loan_rate * trues[~positive_balance]

    if return_all_values:
        return profit
    return float(profit.sum())