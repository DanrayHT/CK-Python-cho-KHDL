import pandas as pd
import numpy as np
from typing import Optional, Union, List, Dict, Literal
from pathlib import Path
import logging
import warnings
pd.set_option('future.no_silent_downcasting', True)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from scipy import stats


class DataPreprocessor:
    """
    Lớp tiền xử lí dữ liệu
    
    Chức năng chính:
    - Đọc dữ liệu từ nhiều định dạng (CSV, Excel, JSON)
    - Tự động phát hiện kiểu dữ liệu (numeric, categorical, datetime)
    - Xử lí giá trị thiếu (missing values)
    - Phát hiện và xử lí ngoại lai (outliers)
    - Chuẩn hóa dữ liệu (normalization/standardization)
    - Mã hóa biến phân loại (encoding)
    - Tạo đặc trưng mới (feature engineering)
    - Xử lý dữ liệu mới cho dự đoán
    """

    def __init__(self, target_column: Optional[str] = None, random_state: int = 42):
        """
        Khởi tạo DataPreprocessor
        
        Parameters:
            target_column (str, optional): Tên cột target. Mặc định None.
            random_state (int): Seed cho random number generator. Mặc định 42.
        """
        self.target_column = target_column
        self.random_state = random_state
        self.data = None
        self.original_data = None
        
        # Lưu trữ thông tin về các loại cột
        self.numeric_columns = []
        self.categorical_columns = []
        self.datetime_columns = []
        
        # Lưu trữ scalers và encoders đã fit
        self.scalers = {}
        self.encoders = {}
        
        # Thiết lập logging
        self._setup_logging()

    def __repr__(self) -> str:
        """
        Returns:
            str: Chuỗi mô tả object
        """
        if self.data is not None:
            return f"DataPreprocessor(shape={self.data.shape}, target='{self.target_column}')"
        return f"DataPreprocessor(target='{self.target_column}', no_data_loaded=True)"

    def _setup_logging(self):
        """
        Thiết lập logging system
        Sử dụng logger chung của module để ghi vào cùng file với model training
        """
        # Sử dụng logger chung từ module thay vì tạo riêng
        self.logger = logging.getLogger(__name__)
        
        # Nếu chưa có handler nào (logging chưa được config), thiết lập basic config
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )


    # ============================================================
    # PHẦN 1: ĐỌC DỮ LIỆU
    # ============================================================

    def load_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Đọc dữ liệu từ file
        Hỗ trợ các định dạng: CSV (.csv), Excel (.xlsx, .xls), JSON (.json)
        Tự động phát hiện kiểu dữ liệu của các cột sau khi load.
        
        Returns:
            pd.DataFrame: DataFrame đã được load
        
        Raises:
            FileNotFoundError: Nếu file không tồn tại
            ValueError: Nếu định dạng file không được hỗ trợ
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File không tồn tại: {file_path}")

            # Đọc dữ liệu dựa vào extension
            ext = file_path.suffix.lower()
            if ext == '.csv':
                self.data = pd.read_csv(file_path)
            elif ext in ['.xlsx', '.xls']:
                self.data = pd.read_excel(file_path)
            elif ext == '.json':
                self.data = pd.read_json(file_path)
            else:
                raise ValueError(f"Định dạng '{ext}' không được hỗ trợ. Hỗ trợ: .csv, .xlsx, .xls, .json")

            # Xóa cột 'id' nếu có (thường không cần cho modeling)
            if 'id' in self.data.columns:
                self.data.drop(columns=['id'], inplace=True)
                self.logger.info("Đã xóa cột 'id'")

            # Lưu bản gốc để có thể reset
            self.original_data = self.data.copy()
            
            # Tự động phát hiện kiểu dữ liệu
            self._detect_column_types()
            
            self.logger.info(f"Đã load dữ liệu thành công: {self.data.shape}")
            return self.data

        except Exception as e:
            self.logger.error(f"Lỗi khi đọc dữ liệu: {e}")
            raise

    def _detect_column_types(self):
        """
        Tự động phát hiện kiểu dữ liệu của các cột
        
        Phân loại các cột thành:
        - numeric_columns: Cột số (int, float)
        - categorical_columns: Cột phân loại (object, string)
        - datetime_columns: Cột thời gian (datetime) 
        """
        if self.data is None:
            return

        self.numeric_columns = []
        self.categorical_columns = []
        self.datetime_columns = []

        for col in self.data.columns:
            # Bỏ qua cột target
            if col == self.target_column:
                continue

            # Kiểm tra datetime columns
            if pd.api.types.is_datetime64_any_dtype(self.data[col]):
                self.datetime_columns.append(col)
                continue

            # Thử detect datetime từ string columns
            if self.data[col].dtype == 'object':
                sample = self.data[col].dropna().head(100)
                if len(sample) >= 10:
                    try:
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore')
                            converted = pd.to_datetime(sample, errors='coerce')
                        
                        # Chỉ coi là datetime nếu ≥80% giá trị convert thành công
                        valid_ratio = converted.notna().sum() / len(sample)
                        if valid_ratio >= 0.8:
                            self.data[col] = pd.to_datetime(self.data[col], errors='coerce')
                            self.datetime_columns.append(col)
                            continue
                    except:
                        pass

            # Phân loại numeric hoặc categorical
            if pd.api.types.is_numeric_dtype(self.data[col]):
                self.numeric_columns.append(col)
            else:
                self.categorical_columns.append(col)

        self.logger.info(f"Phát hiện - Numeric: {len(self.numeric_columns)}, "
                        f"Categorical: {len(self.categorical_columns)}, "
                        f"Datetime: {len(self.datetime_columns)}")

    # ============================================================
    # PHẦN 2: XỬ LÍ GIÁ TRỊ THIẾU (MISSING VALUES)
    # ============================================================

    def handle_missing_values(
            self,
            numeric_strategy: Literal['median', 'mean', 'most_frequent', 'constant'] = 'median',
            categorical_strategy: Literal['most_frequent'] = 'most_frequent',
            fill_value_categorical: str | None = 'missing_category'
        ) -> pd.DataFrame:
            """
            Xử lí giá trị thiếu (missing values)

            Chiến lược xử lý:
            - Cột số (numeric):
                - 'median': Điền bằng trung vị (median)
                - 'mean': Điền bằng trung bình (mean)
                - 'most_frequent': Điền bằng giá trị phổ biến nhất (mode)
                - 'constant': Điền bằng một giá trị cố định (fill_value_numeric)
            - Cột phân loại (categorical):
                - 'most_frequent': Điền bằng giá trị phổ biến nhất (mode)
                - 'constant': Điền bằng một giá trị cố định (fill_value_categorical)

            Sử dụng SimpleImputer từ sklearn

            Args:
                numeric_strategy (Literal): Chiến lược xử lý cho cột số. Mặc định là 'median'.
                categorical_strategy (Literal): Chiến lược xử lý cho cột phân loại. Mặc định là 'most_frequent'.
                fill_value_numeric (float | int | None): Giá trị cố định để điền cho cột số nếu numeric_strategy='constant'. Mặc định là 0.
                fill_value_categorical (str | None): Giá trị cố định để điền cho cột phân loại nếu categorical_strategy='constant'. Mặc định là 'missing_category'.

            Returns:
                pd.DataFrame: DataFrame đã được xử lí missing values
            """
            self.logger.info("Bắt đầu xử lí missing values...")

            # --- Xử lý missing cho cột numeric ---
            if self.numeric_columns:
                # Xác định tham số cho SimpleImputer của cột số
                imputer_params_num = {}
                imputer_params_num['strategy'] = numeric_strategy

                imputer_num = SimpleImputer(**imputer_params_num)
                
                for col in self.numeric_columns:
                    if self.data[col].isnull().any():
                        # Fit và transform chỉ trên cột có giá trị thiếu
                        self.data[[col]] = imputer_num.fit(self.data[[col]]).transform(self.data[[col]])

            # --- Xử lý missing cho cột categorical ---
            if self.categorical_columns:
                # Xác định tham số cho SimpleImputer của cột phân loại
                imputer_params_cat = {}
                imputer_params_cat['strategy'] = categorical_strategy

                if categorical_strategy == 'constant':
                    imputer_params_cat['fill_value'] = fill_value_categorical
                
                imputer_cat = SimpleImputer(**imputer_params_cat)
                
                for col in self.categorical_columns:
                    if self.data[col].isnull().any():
                        # Fit và transform chỉ trên cột có giá trị thiếu
                        # .ravel() để chuyển từ mảng 2D (sau SimpleImputer) về mảng 1D cho Series
                        self.data[col] = imputer_cat.fit(self.data[[col]]).transform(self.data[[col]]).ravel()

            remaining = self.data.isnull().sum().sum()
            self.logger.info(f"Hoàn tất xử lí missing. Còn lại: {remaining}")
            return self.data
    # ============================================================
    # PHẦN 3: PHÁT HIỆN VÀ XỬ LÍ NGOẠI LAI (OUTLIERS)
    # ============================================================

    def detect_outliers_smart(self, col: str, **kwargs) -> List[int]:
            """
            Phát hiện outliers thông minh dựa trên phân bố dữ liệu.
            
            Tự động chọn phương pháp phù hợp:
            - Phân bố lệch nhiều (|skewness| > 2): Isolation Forest
            - Phân bố gần chuẩn (|skewness| < 0.5): Z-score
            - Mặc định: IQR (Interquartile Range)
            
            Args:
                col (str): Tên cột cần phát hiện outliers.
                **kwargs: Các tham số tùy chỉnh:
                    - z_score_threshold (float): Ngưỡng cho Z-score. Mặc định 3.
                    - iqr_multiplier (float): Hệ số nhân cho IQR. Mặc định 1.5.
                    - isolation_contamination (float): Tỷ lệ lây nhiễm (outlier percentage) 
                                                    cho Isolation Forest. Mặc định 0.05.

            Returns:
                List[int]: Danh sách các index của các outliers được phát hiện.
            """
            series = self.data[col].dropna()
            if len(series) < 10:
                return []

            # Lấy các tham số từ kwargs hoặc sử dụng giá trị mặc định
            Z_SCORE_THRESHOLD = kwargs.get('z_score_threshold', 3.0)
            IQR_MULTIPLIER = kwargs.get('iqr_multiplier', 1.5)
            ISO_CONTAMINATION = kwargs.get('isolation_contamination', 0.05)
            
            skewness = series.skew()

            # Case 1: Phân bố lệch nhiều -> dùng Isolation Forest
            # Isolation Forest hiệu quả với dữ liệu phân bố không chuẩn (non-Gaussian)
            if abs(skewness) > 2:
                self.logger.debug(f"Phát hiện outliers cho {col} bằng Isolation Forest.")
                iso = IsolationForest(contamination=ISO_CONTAMINATION, random_state=self.random_state)
                labels = iso.fit_predict(series.values.reshape(-1, 1))
                return list(series.index[labels == -1])

            # Case 2: Phân bố gần chuẩn -> dùng Z-score
            # Z-score (Standard Deviation) giả định phân bố chuẩn (Gaussian)
            if abs(skewness) < 0.5:
                self.logger.debug(f"Phát hiện outliers cho {col} bằng Z-score.")
                z = np.abs(stats.zscore(series))
                return list(series.index[z > Z_SCORE_THRESHOLD])

            # Case 3: Mặc định -> dùng IQR (Interquartile Range)
            # IQR là phương pháp phi tham số (non-parametric), tốt cho dữ liệu bị lệch trung bình
            self.logger.debug(f"Phát hiện outliers cho {col} bằng IQR.")
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - IQR_MULTIPLIER * IQR
            upper = Q3 + IQR_MULTIPLIER * IQR
            mask = (series < lower) | (series > upper)
            return list(series[mask].index)
    
    def handle_outliers(
        self,
        outliers_dict: Dict[str, List[int]],
        strategy: Literal['clip', 'drop', 'impute'] = 'clip',
        impute_value: Union[float, Literal['median']] = 'median'
    ) -> pd.DataFrame:
        """
        Xử lý các outliers đã được phát hiện dựa trên chiến lược đã chọn.

        Args:
            outliers_dict (Dict[str, List[int]]): Dictionary chứa thông tin outliers:
                                                  {'tên_cột': [danh_sách_index_outlier]}.
            strategy (Literal): Chiến lược xử lý outliers ('clip', 'drop', 'impute'). Mặc định là 'clip'.
            impute_value (Union[float, Literal['median']]): Giá trị dùng để thay thế nếu strategy='impute'.
                                                            Mặc định là 'median'.

        Returns:
            pd.DataFrame: DataFrame đã được xử lí outliers.
        """
        df = self.data.copy()
        
        # 1. DROP Strategy
        if strategy == 'drop':
            # Gom tất cả các index outliers từ các cột để loại bỏ các hàng.
            all_outlier_indices = set()
            for col, indices in outliers_dict.items():
                all_outlier_indices.update(indices)
            
            initial_rows = len(df)
            df.drop(list(all_outlier_indices), inplace=True)
            self.logger.info(f"Chiến lược 'drop': Đã loại bỏ {len(all_outlier_indices)} hàng chứa outliers.")
            return df

        # 2. CLIP và IMPUTE Strategies
        for col, indices in outliers_dict.items():
            if indices:
                outlier_series = df.loc[indices, col]
                
                if strategy == 'clip':
                    # Áp dụng lại logic IQR để tìm ngưỡng clip (vì IQR là mặc định)
                    # Nếu outlier được tìm bằng Z-score/Isolation Forest, việc clip có thể dùng ngưỡng tùy ý 
                    # hoặc dùng lại IQR như một phương pháp phi tham số an toàn.
                    # Ở đây ta tái sử dụng logic IQR từ detect_outliers_smart (với IQR_MULTIPLIER mặc định 1.5)
                    
                    series_non_outlier = df[col].drop(index=indices, errors='ignore').dropna()
                    
                    # Nếu series không có đủ dữ liệu để tính Q1, Q3 (ví dụ, toàn bộ là outlier), ta bỏ qua
                    if len(series_non_outlier) < 2:
                         self.logger.warning(f"Không thể clip cho cột '{col}': Không đủ dữ liệu non-outlier.")
                         continue
                         
                    Q1 = series_non_outlier.quantile(0.25)
                    Q3 = series_non_outlier.quantile(0.75)
                    IQR = Q3 - Q1
                    # Giả định IQR_MULTIPLIER mặc định là 1.5 cho clipping, trừ khi có tham số khác được truyền vào lớp
                    IQR_MULTIPLIER = 1.5 # Có thể lấy từ self nếu được lưu trữ
                    lower = Q1 - IQR_MULTIPLIER * IQR
                    upper = Q3 + IQR_MULTIPLIER * IQR
                    
                    # Clip các giá trị outlier
                    df.loc[indices, col] = np.clip(outlier_series, a_min=lower, a_max=upper)
                    self.logger.debug(f"Chiến lược 'clip': Đã clip {len(indices)} outliers trong cột '{col}'.")

                elif strategy == 'impute':
                    impute_val = impute_value
                    if impute_value == 'median':
                        # Tính median từ DỮ LIỆU KHÔNG PHẢI OUTLIER
                        impute_val = df[col].drop(index=indices, errors='ignore').median()
                        if pd.isna(impute_val):
                            # Nếu median không tính được (ví dụ, cột trống), dùng mean
                            impute_val = df[col].drop(index=indices, errors='ignore').mean()
                            
                    # Thực hiện thay thế
                    if not pd.isna(impute_val):
                        df.loc[indices, col] = impute_val
                        self.logger.debug(f"Chiến lược 'impute': Đã thay thế {len(indices)} outliers trong cột '{col}' bằng {impute_val}.")
                    else:
                        self.logger.warning(f"Không thể impute cho cột '{col}': Giá trị thay thế không hợp lệ (NaN).")
                        
                else:
                    self.logger.error(f"Chiến lược xử lí '{strategy}' không hợp lệ. Bỏ qua cột '{col}'.")


        return df
    
    def handle_all_outliers(
            self,
            strategy: Literal['clip', 'drop', 'impute'] = 'clip',
            impute_value: Union[float, Literal['median']] = 'median',
            detection_params: Dict = None
        ) -> pd.DataFrame:
            """
            Tự động phát hiện và xử lý outliers cho TẤT CẢ các cột số.

            Quy trình:
            1. Dùng detect_outliers_smart() để phát hiện outliers cho từng cột số.
            2. Dùng handle_outliers() để xử lý các outliers đã phát hiện bằng chiến lược đã chọn.

            Args:
                strategy (Literal): Chiến lược xử lý outliers ('clip', 'drop', 'impute'). Mặc định là 'clip'.
                impute_value (Union[float, Literal['median']]): Giá trị dùng để thay thế nếu strategy='impute'.
                                                                Mặc định là 'median'.
                detection_params (Dict): Dictionary chứa các tham số tùy chỉnh cho detect_outliers_smart().

            Returns:
                pd.DataFrame: DataFrame đã được xử lí outliers.
            """
            self.logger.info(f"Bắt đầu quy trình phát hiện và xử lí outliers cho {len(self.numeric_columns)} cột số.")

            # 1. Phát hiện Outliers cho tất cả các cột số
            outliers_map: Dict[str, List[int]] = {}
            
            # Thiết lập tham số mặc định nếu không được cung cấp
            detection_params = detection_params if detection_params is not None else {}

            for col in self.numeric_columns:
                # Truyền các tham số tùy chỉnh vào hàm phát hiện thông minh
                indices = self.detect_outliers_smart(col=col, **detection_params)
                
                if indices:
                    outliers_map[col] = indices

            total_outliers = sum(len(v) for v in outliers_map.values())
            self.logger.info(f"Hoàn tất phát hiện. Tìm thấy tổng cộng {total_outliers} outliers trong {len(outliers_map)} cột.")

            if not outliers_map:
                self.logger.info("Không tìm thấy outliers nào cần xử lí. Trả về DataFrame gốc.")
                return self.data

            # 2. Xử lý Outliers
            self.data = self.handle_outliers(
                outliers_dict=outliers_map,
                strategy=strategy,
                impute_value=impute_value
            )

            self.logger.info("Hoàn tất xử lí outliers tổng thể.")
            return self.data

    # ============================================================
    # PHẦN 5: TẠO ĐẶC TRƯNG MỚI (FEATURE ENGINEERING)
    # ============================================================

    def create_polynomial_features(self, columns: Optional[List[str]] = None, degree: int = 2) -> pd.DataFrame:
        """
        Tạo đặc trưng đa thức (polynomial features)
        
        Từ cột x, tạo ra x^2, x^3, ..., x^degree
        Giúp model capture quan hệ phi tuyến (non-linear relationships)
        
        Parameters:
            columns (List[str], optional): Danh sách cột cần tạo polynomial.
                                          Mặc định: 5 cột numeric đầu tiên.
            degree (int): Bậc cao nhất của đa thức. Mặc định 2.
        
        Returns:
            pd.DataFrame: DataFrame với polynomial features mới
        """
        if self.data is None:
            raise ValueError("Chưa có dữ liệu")

        if columns is None:
            columns = self.numeric_columns[:5]  # Giới hạn để tránh quá nhiều features

        for col in columns:
            if col in self.data.columns:
                for d in range(2, degree + 1):
                    new_col_name = f'{col}_pow{d}'
                    self.data[new_col_name] = self.data[col] ** d
                    self.logger.info(f"Tạo feature: {new_col_name}")

        self.logger.info(f"Đã tạo polynomial features bậc {degree} cho {len(columns)} cột")
        self._detect_column_types()
        return self.data

    def create_interaction_features(self, column_pairs: Optional[List[tuple]] = None) -> pd.DataFrame:
        """
        Tạo đặc trưng tương tác (interaction features) giữa các cặp cột
        
        Từ cặp (x, y), tạo ra:
        - x × y (phép nhân)
        - x ÷ y (phép chia)
        - x + y (phép cộng)
        - x - y (phép trừ)
        
        Giúp model capture tương tác giữa các biến.
        
        Parameters:
            column_pairs (List[tuple], optional): Danh sách các cặp (col1, col2).
                                                 Mặc định: auto chọn 3 cặp đầu từ numeric.
        
        Returns:
            pd.DataFrame: DataFrame với interaction features mới
        """
        if self.data is None:
            raise ValueError("Chưa có dữ liệu")

        if column_pairs is None:
            # Tự động chọn một số cặp từ numeric columns
            if len(self.numeric_columns) >= 2:
                column_pairs = [
                    (self.numeric_columns[i], self.numeric_columns[i+1])
                    for i in range(min(3, len(self.numeric_columns)-1))
                ]
            else:
                column_pairs = []

        for col1, col2 in column_pairs:
            if col1 in self.data.columns and col2 in self.data.columns:
                # Phép nhân
                self.data[f'{col1}_x_{col2}'] = self.data[col1] * self.data[col2]
                
                # Phép chia (tránh chia cho 0)
                col2_safe = self.data[col2].replace(0, 1e-6)
                self.data[f'{col1}_div_{col2}'] = self.data[col1] / col2_safe
                
                # Phép cộng
                self.data[f'{col1}_plus_{col2}'] = self.data[col1] + self.data[col2]
                
                # Phép trừ
                self.data[f'{col1}_minus_{col2}'] = self.data[col1] - self.data[col2]
                
                self.logger.info(f"Tạo 4 interaction features cho ({col1}, {col2})")

        self.logger.info(f"Đã tạo interaction features cho {len(column_pairs)} cặp cột")
        self._detect_column_types()
        return self.data

    # ============================================================
    # PHẦN 6: MÃ HÓA BIẾN PHÂN LOẠI (ENCODING)
    # ============================================================

    def encode_categorical(self) -> pd.DataFrame:
        """
        Mã hóa biến phân loại tự động
        
        Chiến lược mã hóa:
        - Binary (2 giá trị unique): Label Encoding
        - Low cardinality (3-10 giá trị): One-Hot Encoding
        - High cardinality (>10 giá trị): Label Encoding
        
        One-Hot Encoding:
        - Sử dụng drop='first' để tránh dummy variable trap
        - handle_unknown='ignore' để xử lý giá trị mới khi inference
        
        Returns:
            pd.DataFrame: DataFrame đã mã hóa
        """
        for col in self.categorical_columns:
            unique_vals = self.data[col].nunique()

            # Binary hoặc High cardinality -> Label Encoding
            if unique_vals <= 2 or unique_vals > 10:
                le = LabelEncoder()
                self.data[col] = le.fit_transform(self.data[col].astype(str))
                self.encoders[col] = le
                self.logger.info(f"Label Encoding: '{col}' ({unique_vals} values)")

            # Low cardinality (3-10) -> One-Hot Encoding
            else:
                ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
                transformed = ohe.fit_transform(self.data[[col]])
                self.encoders[col] = ohe

                # Tạo tên cột mới (bỏ category đầu tiên vì drop='first')
                new_cols = [f"{col}_{cat}" for cat in ohe.categories_[0][1:]]
                for i, new_col in enumerate(new_cols):
                    self.data[new_col] = transformed[:, i]
                
                # Xóa cột gốc
                self.data.drop(columns=[col], inplace=True)
                self.logger.info(f"OneHot Encoding: '{col}' ({unique_vals} values -> {len(new_cols)} cols)")

        # Cập nhật lại danh sách columns
        self._detect_column_types()
        return self.data

    # ============================================================
    # PHẦN 7: CHUẨN HÓA DỮ LIỆU (NORMALIZATION/STANDARDIZATION)
    # ============================================================

    def normalize(self, method: str = 'standard') -> pd.DataFrame:
        """
        Chuẩn hóa dữ liệu số
        
        Các phương pháp:
        - 'standard': StandardScaler - mean=0, std=1 (chuẩn hóa)
        - 'minmax': MinMaxScaler - scale về [0, 1]
        - 'robust': RobustScaler - sử dụng median và IQR (tốt cho outliers)
        
        Parameters:
            method (str): Phương pháp chuẩn hóa. Mặc định 'standard'.
        
        Returns:
            pd.DataFrame: DataFrame đã chuẩn hóa
        
        Note:
            Scalers được lưu trong self.scalers để dùng cho dữ liệu mới
        """
        for col in self.numeric_columns:
            # Tạo scaler theo phương pháp
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            else:
                scaler = RobustScaler()

            # Fit và transform
            self.data[col] = scaler.fit_transform(self.data[[col]])
            self.scalers[col] = scaler

        self.logger.info(f"Đã chuẩn hóa {len(self.numeric_columns)} cột bằng {method.upper()}")
        return self.data

    # ============================================================
    # PHẦN 8: PIPELINE TỰ ĐỘNG
    # ============================================================

    def auto_process_by_data_types(
            self,
            missing_strategy_num: Literal['median', 'mean', 'most_frequent'] = 'median',
            missing_strategy_cat: Literal['most_frequent', 'constant'] = 'most_frequent',
            outlier_strategy: Literal['clip', 'drop', 'impute'] = 'clip',
            detection_params: Dict = None
        ) -> pd.DataFrame:
            """
            Pipeline tự động xử lí dữ liệu dựa trên kiểu dữ liệu với các tham số cấu hình.

            Quy trình:
            1. Xử lý missing values (có thể cấu hình median/mean cho số, mode/constant cho phân loại)
            2. Tạo datetime features (nếu có cột datetime)
            3. Phát hiện và xử lý outliers (cho numeric columns) bằng phương pháp thông minh (có thể cấu hình clip/drop/impute)
            4. Mã hóa categorical columns (auto chọn Label/OneHot)
            5. Chuẩn hóa numeric columns (StandardScaler)

            Args:
                missing_strategy_num (Literal): Chiến lược điền giá trị thiếu cho cột số. Mặc định 'median'.
                missing_strategy_cat (Literal): Chiến lược điền giá trị thiếu cho cột phân loại. Mặc định 'most_frequent'.
                outlier_strategy (Literal): Chiến lược xử lý outliers ('clip', 'drop', 'impute'). Mặc định 'clip'.
                detection_params (Dict): Tham số tùy chỉnh cho detect_outliers_smart().

            Returns:
                pd.DataFrame: DataFrame đã xử lí hoàn chỉnh, sẵn sàng cho modeling
            """
            self.logger.info("Bắt đầu auto preprocessing pipeline...")

            # Bước 1: Xử lý missing values (Cập nhật gọi hàm với tham số mới)
            self.handle_missing_values(
                numeric_strategy=missing_strategy_num,
                categorical_strategy=missing_strategy_cat
            )

            # Bước 2: Tạo datetime features
            if self.datetime_columns:
                self.create_datetime_features()

            # Bước 3: Phát hiện và xử lý outliers (Thay thế bằng hàm tổng quát handle_all_outliers)
            # Lưu ý: Cần đảm bảo self.handle_all_outliers đã được thêm vào lớp.
            if self.numeric_columns:
                
                # Sử dụng hàm tổng quát mới (hoặc logic chi tiết như dưới, nhưng dùng hàm tổng quát sẽ gọn hơn)
                # Dùng logic cũ nhưng cập nhật gọi hàm:
                
                outliers_dict = {}
                # Áp dụng tham số phát hiện (detection_params)
                params = detection_params if detection_params is not None else {}
                
                for col in self.numeric_columns:
                    indices = self.detect_outliers_smart(col=col, **params)
                    if indices:
                        outliers_dict[col] = indices

                total_outliers = sum(len(v) for v in outliers_dict.values())
                
                if total_outliers > 0:
                    self.logger.info(f"Tìm thấy {total_outliers} outliers")
                    # Cập nhật gọi hàm xử lý outliers với chiến lược mới
                    self.handle_all_outliers(strategy=outlier_strategy, detection_params=detection_params)
                    
            # Bước 4: Mã hóa categorical (Giả định self.categorical_columns vẫn được cập nhật sau khi xử lý missing)
            if self.categorical_columns:
                self.encode_categorical()

            # Bước 5: Chuẩn hóa numeric (Giả định self.numeric_columns vẫn hợp lệ)
            # Lưu ý: Nếu bước 3 dùng chiến lược 'drop', các index của self.data đã thay đổi.
            if self.numeric_columns:
                self.normalize(method='standard')

            self.logger.info(f"Hoàn thành preprocessing. Shape: {self.data.shape}")
            return self.data
    # ============================================================
    # PHẦN 9: HÀM TIỆN ÍCH
    # ============================================================

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, target_column: Optional[str] = None):
        """
        Tạo DataPreprocessor từ DataFrame có sẵn
        
        Parameters:
            df (pd.DataFrame): DataFrame đầu vào
            target_column (str, optional): Tên cột target
        
        Returns:
            DataPreprocessor: Instance mới đã được khởi tạo
        """
        preprocessor = cls(target_column=target_column)
        preprocessor.data = df.copy()
        preprocessor.original_data = df.copy()
        preprocessor._detect_column_types()
        return preprocessor

    def get_processed_data(self) -> pd.DataFrame:
        """
        Trả về DataFrame đã được xử lí
        
        Returns:
            pd.DataFrame: Bản copy của data đã xử lí
        
        Raises:
            ValueError: Nếu chưa có dữ liệu
        """
        if self.data is None:
            raise ValueError("Chưa có dữ liệu")
        return self.data.copy()

    def save_processed_data(self, file_path: Union[str, Path], format: str = 'csv'):
        """
        Ghi dữ liệu sau khi xử lý ra file
        
        Parameters:
            file_path (str hoặc Path): Đường dẫn file output
            format (str): Định dạng file ('csv', 'excel', 'json'). Mặc định 'csv'.
        
        Raises:
            ValueError: Nếu chưa có dữ liệu hoặc format không hỗ trợ
        """
        if self.data is None:
            raise ValueError("Chưa có dữ liệu để lưu")

        file_path = Path(file_path)
        if format == 'csv':
            self.data.to_csv(file_path, index=False)
        elif format == 'excel':
            self.data.to_excel(file_path, index=False)
        elif format == 'json':
            self.data.to_json(file_path, orient='records')
        else:
            raise ValueError(f"Format '{format}' không hỗ trợ. Hỗ trợ: csv, excel, json")

        self.logger.info(f"Đã lưu vào {file_path}")

    def preprocess_new_patient(self, new_patient_data: Union[list, dict], processed_df: pd.DataFrame) -> pd.DataFrame:
        """
        Xử lý dữ liệu bệnh nhân mới để dự đoán
        
        Sử dụng encoders và scalers đã được fit từ training data để transform
        dữ liệu mới, đảm bảo consistency với training data.
        
        Parameters:
            new_patient_data (list hoặc dict): Dữ liệu bệnh nhân mới
                - Nếu là list: phải theo đúng thứ tự columns
                - Nếu là dict: key là tên cột, value là giá trị
            processed_df (pd.DataFrame): DataFrame training đã xử lý (để biết columns)
        
        Returns:
            pd.DataFrame: DataFrame đã transform, sẵn sàng cho prediction (1 row)
        
        Raises:
            ValueError: Nếu preprocessor chưa được fit
        """
        if not self.scalers and not self.encoders:
            raise ValueError("Preprocessor chưa được fit. Vui lòng chạy auto_process_by_data_types() trước.")
        
        # Định nghĩa columns cho heart disease dataset
        columns = [
            "age", "sex", "dataset", "cp", "trestbps", "chol", "fbs",
            "restecg", "thalch", "exang", "oldpeak", "slope",
            "ca", "thal", "num"
        ]
        
        # Convert sang DataFrame
        if isinstance(new_patient_data, list):
            new_patient = pd.DataFrame([new_patient_data], columns=columns)
        elif isinstance(new_patient_data, dict):
            new_patient = pd.DataFrame([new_patient_data])
        else:
            new_patient = new_patient_data.copy()
        
        self.logger.info(f"Bắt đầu preprocess bệnh nhân mới: {new_patient.shape}")
        
        # Xóa cột target nếu có
        if 'num' in new_patient.columns:
            new_patient = new_patient.drop(columns=['num'])
        
        # Apply encoders đã fit
        for col, encoder in self.encoders.items():
            if col not in new_patient.columns:
                continue
                
            if isinstance(encoder, LabelEncoder):
                try:
                    new_patient[col] = encoder.transform(new_patient[col].astype(str))
                    self.logger.info(f"Label encoded '{col}'")
                except ValueError as e:
                    # Unknown category -> gán về 0
                    self.logger.warning(f"Unknown value trong '{col}': {e}, gán=0")
                    new_patient[col] = 0
                    
            elif isinstance(encoder, OneHotEncoder):
                try:
                    transformed = encoder.transform(new_patient[[col]])
                    new_cols = [f"{col}_{cat}" for cat in encoder.categories_[0][1:]]
                    
                    for i, new_col in enumerate(new_cols):
                        new_patient[new_col] = transformed[:, i]
                    
                    new_patient = new_patient.drop(columns=[col])
                    self.logger.info(f"OneHot encoded '{col}'")
                except Exception as e:
                    self.logger.warning(f"Lỗi khi encode '{col}': {e}")
        
        # Apply scalers đã fit
        for col, scaler in self.scalers.items():
            if col in new_patient.columns:
                new_patient[col] = scaler.transform(new_patient[[col]])
                self.logger.info(f"Scaled '{col}'")
        
        # Đảm bảo có đầy đủ columns như training data
        X_train_columns = processed_df.drop('num', axis=1, errors='ignore').columns.tolist()
        
        # Thêm missing columns với giá trị 0
        missing_cols = set(X_train_columns) - set(new_patient.columns)
        for col in missing_cols:
            new_patient[col] = 0
            if col in self.scalers:
                new_patient[col] = self.scalers[col].transform(new_patient[[col]])
        
        # Sắp xếp columns theo thứ tự training data
        new_patient = new_patient[X_train_columns]
        
        self.logger.info(f"Hoàn thành preprocess. Shape: {new_patient.shape}")
        return new_patient
