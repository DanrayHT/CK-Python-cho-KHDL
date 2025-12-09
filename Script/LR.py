from .model import BaseModel
import numpy as np
import pandas as pd
from typing import Union, Optional
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

class LogisticRegressionModel(BaseModel):
    def __init__(self, X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
                y: Optional[Union[pd.Series, np.ndarray]] = None,
                test_size: float = 0.3,
                random_state: int = 42,
                C: float = 1.0,
                penalty: str = "l2",
                solver: str = "lbfgs",
                max_iter: int = 1000,
                **kwargs):
        """
        Khởi tạo thêm các thuộc tính cần thiết
        Args:
            X: dữ liệu được truyền vào
            y: nhãn
            test_size: tỷ lệ tập test
            random_state: seed khi split dữ liệu
            C: Độ mạnh regularization
            penalty: Loại regularization sử dụng. Hỗ trợ:
                - "l2": ổn định, phổ biến nhất
            solver (str):
                Thuật toán tối ưu để huấn luyện Logistic Regression.
                - "lbfgs": đa dụng, hỗ trợ multi-class (mặc định)
                max_iter (int):Số vòng lặp tối đa để thuật toán hội tụ.
            **kwargs:Bất kỳ tham số phụ nào khác của sklearn LogisticRegression,
        """
        super().__init__(X=X, y=y, test_size=test_size, random_state=random_state, name="LogisticRegression")
        self._init_params = {"C": C, "penalty": penalty, "solver": solver, "max_iter": max_iter}
        self._init_params["random_state"] = self._random_state
        self._init_params.update(kwargs)


    def build_model(self):
        self._model = LogisticRegression(**self._init_params)
        return self
    
    def plot_feature_importance(self):
        coefs = self._model.coef_[0]
        plt.figure(figsize=(12, 6))
        plt.barh(self._X.columns, coefs)
        plt.title(f"Logistic RegressionModel")
        plt.xlabel("Coefficient Value")
        plt.tight_layout()
        plt.show()
        return
