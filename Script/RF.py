from .model import BaseModel
import numpy as np
import pandas as pd
from typing import Union, Optional
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

class RandomForestModel(BaseModel):
    def __init__(self, X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
                y: Optional[Union[pd.Series, np.ndarray]] = None,
                test_size: float = 0.3,
                random_state: int = 42,
                n_estimators: int = 100,
                max_depth: Optional[int] = None,
                 **kwargs):
        """
        Khởi tạo thêm các thuộc tính cần thiết
        Args:
            X: dữ liệu được truyền vào
            y: nhãn
            test_size: tỷ lệ tập test
            random_state: seed khi split dữ liệu
            n_estimators: Số lượng cây trong rừng
            max_depth: độ sâu tối đa của cây
            **kwargs:Bất kỳ tham số phụ nào khác của mô hình, được gộp chung vào self._init_params.
        """
        super().__init__(X=X, y=y, test_size=test_size, random_state=random_state, name="RandomForest")
        self._init_params = {"n_estimators": n_estimators, "max_depth": max_depth}
        self._init_params["random_state"] = self._random_state
        self._init_params.update(kwargs)

    def build_model(self):
        """Hàm xây dựng mô hình
        """
        self._model = RandomForestClassifier(**self._init_params)
        return self
    
    def plot_feature_importance(self):
        """Hàm vẽ biểu đồ thể hiện độ quan trọng của từng đặc trung dữ liệu
        """
        importances = self.feature_importances()
        plt.figure(figsize=(12, 6))
        plt.barh(self._X.columns, importances)
        plt.title(f"Random Forest")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.show()
        return
