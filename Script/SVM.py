from .model import BaseModel
import numpy as np
import pandas as pd
from typing import Union, Optional
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

class SVMModel(BaseModel):
    def __init__(self, X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
                y: Optional[Union[pd.Series, np.ndarray]] = None,
                test_size: float = 0.3,
                random_state: int = 42,
                C: float = 1.0,
                kernel: str = "rbf",
                gamma: Union[str, float] = "scale",
                probability: bool = True,
                **kwargs):
        super().__init__(X=X, y=y, test_size=test_size, random_state=random_state, name="SVM")
        self._init_params = {"C": C, "kernel": kernel, "gamma": gamma, "probability": probability}
        self._init_params["random_state"] = self._random_state
        self._init_params.update(kwargs)
        """
        Khởi tạo thêm các thuộc tính cần thiết
        Args:
            X: dữ liệu được truyền vào
            y: nhãn
            test_size: tỷ lệ tập test
            random_state: seed khi split dữ liệu
            C: Độ mạnh regularization
            kernel: loại kernel mô hình dùng
            gamma: thuộc tính điều khiển độ rộng
            probability: cho phép tính predict_proba
            **kwargs:Bất kỳ tham số phụ nào khác của mô hình, được gộp chung vào self._init_params.
        """

    def build_model(self):
        """
        hàm xây dựng mô hình
        """
        self._model = SVC(**self._init_params)
        return self
    
    def plot_feature_importance(self):
        if not hasattr(self._model, "support_"):
            raise RuntimeError("Model must be fitted before calling plot(). Please fit the model first.")
    
        X_np = self._X.values if isinstance(self._X, pd.DataFrame) else self._X
        r = permutation_importance(self._model, X_np, self._y, n_repeats=15, random_state=42)
    
        plt.figure(figsize=(12, 6))
        plt.title(f"SVM Model")
        plt.barh(self._X.columns if isinstance(self._X, pd.DataFrame) else range(X_np.shape[1]), r.importances_mean)
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.show()
