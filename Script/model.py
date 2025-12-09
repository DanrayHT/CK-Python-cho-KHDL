import pandas as pd
import numpy as np
import os
from datetime import datetime
import joblib
from typing import Union, Dict, Any, Callable, List, Optional
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import shap

# hàm hỗ trợ
def _to_numpy(X: Union[pd.DataFrame, pd.Series, np.ndarray]) -> np.ndarray:
  if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
    return X.values
  return np.asarray(X)


def _ensure_dir(path: str):
  d = os.path.dirname(path)
  if d:
    os.makedirs(d, exist_ok=True)
# Exception khi mô hình chưa huấn luyện
class NotFittedError(Exception):
    """Raised khi mô hình chưa được huấn luyện"""
    pass

# Exception khi metric không tồn tại
class MetricNotAvailable(Exception):
    """Raised khi metric không hỗ trợ"""
    pass
#
class NotHasAttr(Exception):
    """ Raised Khi không có thuộc tính mong muốn"""
    pass

class BaseModel:
    """
    Class chung cho cả 4 mô hình với các tính năng như trên
    """
    def __init__(
        self,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        test_size: float = 0.3,
        random_state: int = 42,
        name: str = "BaseModel",
    ) -> None:
        """
        Hàm khởi tạo thuộc tính cho lớp

        Args:
            X: dữ liệu đặc trưng ban đầu. truyền vào để mô hình học
            y: nhãn. Dùng để train mô hình
            test_size: tỉ lệ tập test
            random_state: đảm bảo chia tập train/test ổn định
            name: tên mô hình
        """
        
        self.name = name
        self._random_state = random_state
        self._test_size = test_size

        self._X = X
        self._y = y
        self.__X_train = None
        self.__X_test = None
        self.__y_train = None
        self.__y_test = None

        self._model = None


        self.__is_fitted = False
        self._meta: Dict[str, Any] = {
            "created_at": datetime.now(),
            "trained_at": None,
            "best_params": None,
            }

    # Thực hiện các tính năng chung
    def split_data(self, X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
                    y: Optional[Union[pd.Series, np.ndarray]] = None,
                    test_size: Optional[float] = None,
                    random_state: Optional[int] = None) -> None:
        """
        Chia data thành tập train/ test
        Nếu X, y không được truyền vào, hàm sẽ dùng self.X, self.y
        Args:
            X: dữ liệu đặc trưng ban đầu. truyền vào để mô hình học
            y: nhãn. Dùng để train mô hình
            test_size: tỉ lệ tập test
            random_state: đảm bảo chia tập train/test ổn định
        Return:
            tập X_train, X_test, y_train, y_test
        Raises:
            ValueError: Khi X,y Trống trong cả đối tượng và khi gọi hàm
        """
        if X is None:
            X = self._X
        if y is None:
            y = self._y
        if X is None or y is None:
            raise ValueError("X,y phải được cung cấp hoặc khởi tạo tại __init__")

        ts = self._test_size if test_size is None else test_size
        rs = self._random_state if random_state is None else random_state

        X_np = _to_numpy(X)
        y_np = _to_numpy(y)

        self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(
            X_np, y_np, test_size=ts, random_state=rs, stratify=y_np if len(np.unique(y_np)) > 1 else None
        )


    def build_model(self):
        """
        hàm này sẽ được triển khai trong lớp con
        Raises:
            NotImplementedError: Gọi nhầm đối tượng của class cha
        """
        raise NotImplementedError("Hàm này cần được gọi trong class Con")

    def fit(self, X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
            y: Optional[Union[pd.Series, np.ndarray]] = None,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            **fit_kwargs) -> "BaseModel":
        """
        hàm chạy mô hình.
        Args:
            X: dữ liệu đặc trưng được truyền vào dùng để train mô hình
            y: nhãn dùng để train mô hình
            X_val: các dữ liệu đặc trưng dùng để kiểm định
            y_val: các nhãn dùng để kiểm định
            **fix_kwargs: các tham số cần thiết để train mô hình
        Return:
            trả về đối tượng mô hình (self)
        Raises:
            ValueError: thiếu tập X, y trong dối đượng và khi truyền hàm
        """
        if X is not None and y is not None:
            self._X = X
            self._y = y

        if self.__X_train is None or self.__y_train is None:
            self.split_data()

        if self.__X_train is None or self.__y_train is None:
            raise ValueError("Không tìm thấy data huấn luyện")


        if self._model is None:
            self.build_model()


        self._model.fit(self.__X_train, self.__y_train, **fit_kwargs)
        self.__is_fitted = True
        self._meta["trained_at"] = datetime.now()


        if X_val is not None and y_val is not None:
            try:
                X_val_np = _to_numpy(X_val)
                score = self.evaluate(X_val_np, y_val)
                self._meta["val_score"] = score
            except Exception:
                pass

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        hàm dự đoán nhãn y dựa theo X được truyền vào
        Args:
            X: các dữ liệu đặc trưng được truyền vào.
        Return:
            kết quả dự đoán sau khi train mô hình
        Raises:
            NotFittedError: chưa huấn luyện mô hình
        """
        if not self.__is_fitted:
            raise NotFittedError(f"{self.name} Chưa được huấn luyện.")
        X_np = _to_numpy(X)
        return self._model.predict(X_np)


    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        hàm dự đoán nhãn y dựa theo X được truyền vào
        Args:
            X: các dữ liệu đặc trưng được truyền vào.
        Return:
            xác suất kết quả dự đoán sau khi train mô hình
        Raises:
            NotFittedError: chưa huấn luyện mô hình
            NotImplementedError: dùng hàm này nhầm mô hình( một vài mô hình ko hỗ trợ .predict_proba)
        """
        if not self.__is_fitted:
            raise NotFittedError("Mô hình chưa được huấn luyện")
        if not hasattr(self._model, "predict_proba"):
            raise NotImplementedError("predict_proba không hỗ trợ mô hình này")
        X_np = _to_numpy(X)
        return self._model.predict_proba(X_np)


    def evaluate(self, X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
                y: Optional[Union[pd.Series, np.ndarray]] = None,
                metrics: Optional[List[Callable]] = None) -> Dict[str, float]:
        """
        Hàm dùng để tính các metric mô hình
        Args:
            X: dữ liệu đặc trưng truyền vào dùng để tính metrics
            y: Nhãn dùng để tính metrics
            metrics: các metrics tùy chỉnh nếu người dùng muốn tính thêm
        Return:
            Các giá trị metric default: Accuracy, Precision, Recall, F1, ... ( thêm các metrics tùy chỉnh nếu có)
        Raises:
            ValueError: X_test, y_test không được truyền vào
            NotFittedError: Mô hình chưa train
        """
        if X is None or y is None:
            if self.__X_test is None or self.__y_test is None:
                raise ValueError("No test data available. Provide X and y or call split_data() earlier.")
            X = self.__X_test
            y = self.__y_test
        X_np = _to_numpy(X)
        y_np = _to_numpy(y)

        if not self.__is_fitted:
            raise NotFittedError("Model not fitted yet")

        y_pred = self._model.predict(X_np)
        results: Dict[str, float] = {}
        # metrics mặc định
        results["accuracy"] = float(accuracy_score(y_np, y_pred))
        try:
            results["precision"] = float(precision_score(y_np, y_pred, zero_division=0, average="binary" if len(np.unique(y_np))==2 else "macro"))
        except Exception:
            results["precision"] = float(precision_score(y_np, y_pred, average="macro", zero_division=0))
        try:
            results["recall"] = float(recall_score(y_np, y_pred, zero_division=0, average="binary" if len(np.unique(y_np))==2 else "macro"))
        except Exception:
            results["recall"] = float(recall_score(y_np, y_pred, average="macro", zero_division=0))
        try:
            results["f1"] = float(f1_score(y_np, y_pred, zero_division=0, average="binary" if len(np.unique(y_np))==2 else "macro"))
        except Exception:
            results["f1"] = float(f1_score(y_np, y_pred, average="macro", zero_division=0))

        if len(np.unique(y_np)) == 2 and hasattr(self._model, "predict_proba"):
            try:
                y_proba = self._model.predict_proba(X_np)[:, 1]
                results["roc_auc"] = float(roc_auc_score(y_np, y_proba))
            except Exception:
                pass

        # mectrics thêm (nếu có)
        if metrics is not None:
            for m in metrics:
                name = getattr(m, "__name__", str(m))
                try:
                    results[name] = float(m(y_np, y_pred))
                except Exception:
                    try:
                        results[name] = float(m(y_np, self._model.predict_proba(X_np)))
                    except Exception:
                        results[name] = None
        return results

    def optimize_params(self,
            param_grid: Dict[str, List[Any]],
            search: str = "grid",
            cv: int = 3,
            scoring: Optional[str] = None,
            n_iter: int = 20,
            n_jobs: int = 1,
            verbose: int = 1) -> Dict[str, Any]:
        """
        Hàm tối ưu siêu tham số, sau đó update model
        Args:
            param_grid: danh sách các siêu tham số
            search: phương pháp tìm, mặc định là GridSearchCV
            cv: số lượng fold trong cross-validation
            scoring: metrics dùng để đánh giá
            n_iter: số lượng tổ hợp thử ( chỉ dùng chi RandomizedSearchCV)
            n_jobs: số CPU chạy song song
            verbose: hiển thị log khi tìm tham số
                0: im lặng
                1: cơ bản
                2: chi tiết
        Return:
            Tổ hợp siêu tham số tốt nhất + điểm trung bình trên cv tương ứng
        Raises:
            NotHasAttr: model không hỗ trợ tìm kiếm
            ValueError: Chưa chia tập X_train, y_train

        """
        if self._model is None:
            self.build_model()
        if not hasattr(self._model, "get_params"):
            raise NotHasAttr("Mô hình không hỗ trọ")

        X_train = self.__X_train
        y_train = self.__y_train
        if X_train is None or y_train is None:
            raise ValueError("Dữ liệu X,y chưa được truyền vào ( hoặc chưa gọi split_data())")

        if search == "grid":
            searcher = GridSearchCV(self._model, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs, verbose=verbose)
            searcher.fit(X_train, y_train)
        else:
            searcher = RandomizedSearchCV(self._model, param_distributions=param_grid, n_iter=n_iter, cv=cv, scoring=scoring, n_jobs=n_jobs, verbose=verbose)
            searcher.fit(X_train, y_train)


        self._model = searcher.best_estimator_
        self._meta["best_params"] = searcher.best_params_
        self._meta["best_score"] = float(searcher.best_score_)
        self.__is_fitted = True
        return {"best_params": searcher.best_params_, "best_score": float(searcher.best_score_)}


    def cross_validate(self, cv: int = 5, scoring: Optional[str] = None, n_jobs: int = 1) -> Dict[str, Any]:
        """
        hàm đánh giá mô hình trên toàn bộ dữ liệu
        Args:
            cv: số lượng fold
            scoring: metrics dùng để đánh giá
            n_jobs: số cpu core dùng khi tính score
        Return:
            các giá trị score, trung bình và ssoj lệch chuẩn
        Raises:
            valueError: các giá trị X,y chưa được truyền vào
        """
        if self._model is None:
            self.build_model()
        if self._X is None or self._y is None:
            raise ValueError("Provide X and y at init or before calling cross_validate()")
        X_np = _to_numpy(self._X)
        y_np = _to_numpy(self._y)
        scores = cross_val_score(self._model, X_np, y_np, cv=cv, scoring=scoring, n_jobs=n_jobs)
        return {"cv_scores": scores.tolist(), "cv_mean": float(np.mean(scores)), "cv_std": float(np.std(scores))}



    def feature_importances(self) -> Optional[pd.Series]:
        """
        Hàm dùng để tìm các cột quan trọng nhất. Ảnh hưởng nhiều nhất đến kết quả
        Return:
            trả về Series chứa tên các feature quan trong và index
        Raises:
            NotFittedError: mô hình chưa được train
        """
        if not self.__is_fitted:
            raise NotFittedError("Mô hình chưa được huấn luyện")
        if hasattr(self._model, "feature_importances_"):
            vals = self._model.feature_importances_
            return pd.Series(vals)
        # trường hợp dành cho xgboots
        if hasattr(self._model, "get_booster"):
            try:
                booster = self._model.get_booster()
                fmap = booster.get_score(importance_type="gain")
                return pd.Series(fmap)
            except Exception:
                return None
        return None


    def explain(self, X_sample: Union[pd.DataFrame, np.ndarray], n_samples: int = 100) -> Optional[Any]:
        """
        Tạo SHAP để giải thích mô hình
        Args:
            X_sample: Dữ liệu đặc trưng mà ta muốn giải thích
            nsaples: số quan sát cần giải thích
        Return:
            shap_values hoặc none
        Raises:
            NotFittedError: mô hình chưa được huấn luyện
        """
        if not self.__is_fitted:
            raise NotFittedError("Mô hình chưa được huấn luyện")
        X_np = _to_numpy(X_sample)
        X_np = X_np[:n_samples]

        explainer = None
        try:
            if hasattr(self._model, "predict_proba"):
                explainer = shap.Explainer(self._model.predict_proba, X_np)
            else:
                explainer = shap.Explainer(self._model.predict, X_np)
            shap_values = explainer(X_np)
            return shap_values
        except Exception as e:
            print("SHAP giải thích thất bại:", e)
            return None


    def save_model(self, path: str) -> None:
        """
        Hàm dùng để lưu lại mô hình
        Args:
            path: tên thư mục
        Return:
            đối tượng mô hình
        """
        _ensure_dir(path)
        payload = {
            "meta": self._meta,
            "name": self.name,
            "random_state": self._random_state,
            "test_size": self._test_size,
            "model": self._model,
            "is_fitted": self.__is_fitted,
        }
        joblib.dump(payload, path)

    @classmethod
    def load_model(cls, path: str) -> "BaseModel":
        payload = joblib.load(path)
        inst = cls()
        inst._meta = payload.get("meta", {})
        inst.name = payload.get("name", inst.name)
        inst._random_state = payload.get("random_state", inst._random_state)
        inst._test_size = payload.get("test_size", inst._test_size)
        inst._model = payload.get("model", inst._model)
        inst.__is_fitted = payload.get("_is_fitted", inst.__is_fitted)
        return inst


    def is_fitted(self) -> bool:
        """
        hàm kiểm tra mô hình đẫ được huấn liện hay chưa
        """
        return bool(self.__is_fitted)

    def get_params(self) -> Dict[str, Any]:
        """
        lấy các thông số khi huấn luyện 1 mô hình.
        Return:
            tên + giá trị của các thông số khi huấn luyện
        """
        if self._model is None:
            return {}
        return self._model.get_params()
    def get_name(self) -> str:
        """
        lấy tên mô hình
        """
        return self.name

    def set_params(self, **params) -> "BaseModel":
        """
        hàm dùng để cài các thông số khi huấn luyện mô hình
        """
        if self._model is None:
            self.build_model()
        self._model.set_params(**params)
        return self

    def summary(self) -> Dict[str, Any]:
        """
        return: thông tin mô hình
        """
        return {
            "name": self.name,
            "is_fitted": self.__is_fitted,
            "meta": self._meta,
            "model_params": self.get_params(),
        }
    def print_summary(self):
        """
        hàm hỗ trợ in chô dễ nhìn
        """
        summary_cls = self.summary()
        meta_cls = self._meta
        print("Các thông tin về mô hình")
        for k, v in summary_cls.items():
            if k == "meta":
                continue
            if k == "model_params":
                print(k,":")
                for key, value in v.items():
                    print("\t",key, ":", value)
                continue
            print(k, ":", v)
        for k,v in meta_cls.items():
            print(k, ":", v)


    def plot_confusion_matrix(self):
        if not self.__is_fitted:
            raise NotFittedError("Mô hình chưa được huấn luyện")

        y_pred = self._model.predict(self.__X_test)

        ConfusionMatrixDisplay.from_predictions(self.__y_test, y_pred)
        plt.title(f"Confusion Matrix - {self.name}")
        plt.show()

    def plot_roc_pr(self):
        """
        Vẽ ROC Curve và Precision–Recall Curve cho model hiện tại.
        """
        if not self.is_fitted():
            raise NotFittedError("Mô hình chưa được huấn luyện")

        # Lấy scores
        y_scores = self._get_model_scores(self._model, self.__X_test)
        y_true = self.__y_test

        plt.figure(figsize=(12,5))

        # === ROC Curve ===
        plt.subplot(1,2,1)
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{self.name} (AUC={roc_auc:.3f})")
        plt.plot([0,1],[0,1],'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")

        # === Precision–Recall Curve ===
        plt.subplot(1,2,2)
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
        plt.plot(recall, precision, label=f"{self.name} (AP={ap:.3f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision–Recall Curve")
        plt.legend(loc="lower left")

        plt.tight_layout()
        plt.show()

    def _get_model_scores(self, model, X):
        """
        Trả về xác suất dự đoán cho positive class
        """
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:,1]
        elif hasattr(model, "decision_function"):
            return model.decision_function(X)
        else:
            raise ValueError(f"Model {model} không hỗ trợ predict_proba hoặc decision_function")