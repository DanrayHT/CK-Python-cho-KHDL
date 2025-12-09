from .model import BaseModel, NotFittedError, MetricNotAvailable
from typing import List

class ModelSelector:
    def __init__(self,
                 models: List[BaseModel],
                 metric: str = "accuracy",
                 ):
        """
        Khởi tạo các mô hình để so sánh ( mô hình bắt buộc phải huấn luyện trước)
        Args:
            models: Danh sách các mô hình so sánh
            metric: chỉ số so sánh

        """
        self._models = models
        self._metric = metric
        self._result = []
        self._best_model = None
    def check_fitted(self):
        """
        Hàm kiểm tra xem mô hình đã huấn luyện hay chưa
        Raise:
            NotFittedError: Mô hình chưa huấn luyện
        """
        for i in self._models:
            if not i.is_fitted():
                raise NotFittedError(f"Mô hình {i.get_name()} chưa được huấn luyện")

    def summary(self) -> list:
        """
        hàm so sánh + đưa ra kết quả ~ set self.result
        Return:
            self.result
        Raise:
            MetricNotAvailable: Nhập metric sai
        """
        self.check_fitted()
        self._result = []
        for i in self._models:
            eva = i.evaluate()
            if self._metric == "accuracy":
                score = eva["accuracy"]
                self._result.append({"name": i.get_name(),
                     "score": score,
                     "model": i})
            elif self._metric == "f1":
                score = eva["f1"]
                self._result.append({"name": i.get_name(),
                     "score": score,
                     "model": i})
            elif self._metric == "precision":
                score = eva["precision"]
                self._result.append({"name": i.get_name(),
                     "score": score,
                     "model": i})
            elif self._metric == "recall":
                score = eva["recall"]
                self._result.append({"name": i.get_name(),
                     "score": score,
                     "model": i})
            else:
                raise MetricNotAvailable("Metric không tồn tại trong mô hình, hãy kiểm tra lại input")
        self._result.sort(key = lambda x: x["score"], reverse = True)
    def set_best_model(self):
        """
        Hàm so sánh để tìm mô hình tốt nhất
        """
        if not self._result:
            self.summary()
        best_score = max(self._result, key = lambda x: x["score"])["score"]
        self._best_model = [m for m in self._result if m["score"] == best_score]

    def print_result(self):
        """
        Hàm in summary và result
        """
        if not self._result :
            self.summary()
        if not self._best_model:
            self.set_best_model()
        print("Metric được chọn:", self.get_metric())
        for r in self._result:
            print("tên mô hình:", r["name"],".\n Chỉ số: ", r["score"])
        print("Mô hình tốt nhất là: ")
        for m in self.get_best_model():
            print(" ", m["name"], "=>", m["score"])

    def get_metric(self) -> str:
        """
        Hàm trả về metric
        """
        return self._metric
    def get_best_model(self) -> list:
        """
        Hàm trả về best_model
        """
        return self._best_model