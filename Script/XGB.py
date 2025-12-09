from .model import BaseModel
import numpy as np
import pandas as pd
from typing import Union, Optional
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

class XGBoostModel(BaseModel):
    def __init__(self, X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
                y: Optional[Union[pd.Series, np.ndarray]] = None,
                test_size: float = 0.3,
                random_state: int = 42,
                n_estimators: int = 100,
                learning_rate: float = 0.1,
                max_depth: int = 6,
                **kwargs):
        """
        Khởi tạo thêm các thuộc tính cần thiết
        Args:
            X: dữ liệu được truyền vào
            y: nhãn
            test_size: tỷ lệ tập test
            random_state: seed khi split dữ liệu
            n_estimators: Số lượng boosting rounds( cây trong xboots)
            learning_rate: tốc độ học
            max_depth: độ sâu tối đa của cây
            **kwargs:Bất kỳ tham số phụ nào khác của mô hình, được gộp chung vào self._init_params.
        """
        super().__init__(X=X, y=y, test_size=test_size, random_state=random_state, name="XGBoost")
        self._init_params = {"n_estimators": n_estimators, "learning_rate": learning_rate, "max_depth": max_depth}
        self._init_params["random_state"] = self._random_state
        self._init_params.update(kwargs)

    def build_model(self):
        self._model = XGBClassifier(eval_metric="logloss", **self._init_params)
        return self
    
    def plot_feature_importance(self):
        try:
            booster = self._model.get_booster()
            scores = booster.get_score(importance_type="gain")
            # Align keys f0,f1,f2... với feature names
            importances = [scores.get(f"f{i}", 0) for i in range(len(self._X.columns))]

            plt.figure(figsize=(12, 6))
            plt.barh(self._X.columns, importances)
            plt.title(f"XGBoost Feature Importance (Gain)")
            plt.tight_layout()
            plt.show()
            return
        except Exception:
            pass