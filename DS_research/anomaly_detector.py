from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np

class AnomalyDetector:
    """Детектор аномалий методом IsolationForest"""
    def __init__(self, contamination=0.05):
        self.model = IsolationForest(contamination=contamination, random_state=42)

    def fit(self, X):
        self.model.fit(X)
        return self
    
    def detect(self, X):
        preds = self.model.predict(X)
        anomaly_mask = (preds == -1)
        return anomaly_mask
    
    def remove_anomalies(self, X):
        mask = self.detect(X)
        X_clean = X.loc[~mask]
        return X_clean, mask
    
    @classmethod
    def process(cls, X,  **kwargs):
        X_copy = X.copy()
        X_copy.drop('date', inplace=True, axis=1)
        det = cls(**kwargs)
        det.fit(X_copy)
        clean_data_iforest , mask = det.remove_anomalies(X_copy)
        return clean_data_iforest , mask , det
    
class RealTimeAnomalyDetector:
    def __init__(self, detector):
        # detector - экземпляр AnomalyDetector, уже обученный на исторических данных
        self.detector = detector

    def predict_new_value(self, X: pd.DataFrame, y) -> bool:
        """
        возвращает True, если новое значение - аномалия, иначе False
        """
        X['balance'] = y
        X.drop('date', inplace=True, axis=1)
        # predict дает 1 (норма) или -1 (аномалия)
        pred = self.detector.model.predict(X)
        return pred[0] == -1
    
