# Dự đoán Bệnh Tim Mạch (Heart Disease Prediction)

Dự án này là bài tập cuối kỳ môn **Lập trình Python cho Khoa học Dữ liệu**, tập trung vào việc xây dựng mô hình học máy để dự đoán nguy cơ mắc bệnh tim mạch.

---

## 1. Thông tin thành viên

| Họ và tên | Mã số sinh viên |
|:---|:---:|
| Đỗ Lê Nguyên Đan | 23280044 |
| Lương Lê Công Hạnh | 23280057 |
| Âu Dương Khả | 23280063 |

---

## 2. Mục tiêu bài toán

Xây dựng một mô hình **phân loại (Classification)** nhằm dự đoán khả năng mắc bệnh tim mạch của bệnh nhân dựa trên các đặc trưng sức khoẻ lâm sàng.

* **Input:** Các chỉ số sức khỏe như tuổi, giới tính, huyết áp, cholesterol, kết quả điện tâm đồ, nhịp tim tối đa, v.v.
* **Output:** Dự đoán nhị phân (0: Không có nguy cơ, 1: Có nguy cơ mắc bệnh).
* **Ý nghĩa:** Hỗ trợ các bác sĩ và chuyên gia y tế trong việc sàng lọc và phát hiện sớm nguy cơ bệnh, từ đó đưa ra phác đồ điều trị kịp thời.

---

## 3. Cấu trúc thư mục

Dự án được tổ chức theo cấu trúc sau:

```text
CK-Python-cho-KHDL/
├── Data/                   # Chứa dữ liệu dataset
├── Docs/                   # Tài liệu báo cáo, slide thuyết trình
├── Script/                 # Mã nguồn chính (Source code)
├── configs/                # Chứa các file cấu hình)
├── models_all/             # Chứa các model đã save sau khi chạy với random state 42
├── models_data/            # Chứa các chỉ số của model sau khi chạy với random state 42
├── notebook/               # Jupyter Notebooks cho kết quả cũng như các plot
├── main.py                 # File main để chạy dự đoán
├── requirements.txt        # Các thư viện cần thiết
└── README.md               # Thông tin dự án
````

-----

## 4. Cài đặt

Yêu cầu: **Python 3.8+**

1.  **Clone repository:**

    ```bash
    git clone https://github.com/DanrayHT/CK-Python-cho-KHDL.git
    cd CK-Python-cho-KHDL
    ```

2.  **Cài đặt thư viện:**

    ```bash
    pip install -r requirements.txt
    ```

-----

## 5. Cách sử dụng (Usage)

Hệ thống sử dụng `configparser` để quản lý tham số và `argparse` để cho phép tùy chỉnh nhanh qua dòng lệnh.

### 5.1. File cấu hình (`configs/default.ini`)

Bạn có thể thay đổi các tham số mặc định (đường dẫn dữ liệu, cách xử lý thiếu, thông tin bệnh nhân test) tại đây:

```ini
[PREPROCESSING]
input_file = Data/heart_disease_uci.csv
random_state = 42
missing_strategy_num = median
missing_strategy_cat = most_frequent
outlier_strategy = clip
detection_params_z_score_threshold = 2.5

[PATIENT_INFO]
# Thông tin bệnh nhân mới: [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
patient_data = 37, "Male", "Cleveland", "non-anginal", 130.0, 250.0, False, "normal", 187.0, False, 3.5, "downsloping", 0.0, "normal"

[MODEL_SELECTION]
metric = recall

[OUTPUT]
result_name = models_data/experiment_results
format = csv
```

### 5.2. Chạy chương trình

Chạy file `main.py` từ thư mục gốc của dự án. Chương trình sẽ tự động:

1.  Load và tiền xử lý dữ liệu.
2.  Huấn luyện 4 model (LR, SVM, RF, XGB).
3.  Lưu model vào thư mục `models_all/`.
4.  Chọn model có chỉ số tốt nhất (ví dụ: Recall).
5.  Dự đoán kết quả cho bệnh nhân được khai báo trong config.

**Lệnh chạy cơ bản:**

```bash
python main.py
```

**Kết quả hiển thị:**

```text
Đã đọc cấu hình từ file: configs/default.ini
Dự đoán: **Không bệnh**
Xác suất dự đoán: Không bệnh = 69.09%, Có bệnh = 30.91%
```

### 5.3. Tùy chỉnh qua dòng lệnh (CLI Arguments)

Bạn có thể ghi đè các tham số trong file config bằng các cờ (flags) sau:

  * `--config`: Đường dẫn file config khác.
  * `--input_file`: Đường dẫn file dataset.
  * `--random_state`: Chọn seed để có thể tái tạo kết quả (`int` hoặc `None` để hoàn toàn ngẫu nhiên).
  * `--missing_strategy_num`: Cách xử lý biến numeric (`mean`, `median`, `most_frequent`).
  * `--missing_strategy_cat`: Cách xử lý biến category (`most_frequent`, `constant`).
  * `--detection_params`: Ngưỡng Z-score cho phát hiện outlier. (`float`)
  * `--outlier_strategy`: Cách xử lý outlier (`clip`, `drop`, `impute`).
  * `--patient_info`: Nhập trực tiếp thông tin bệnh nhân mới (dạng chuỗi list).
  * `--result_name`: Thay đổi đường dẫn và tên file lưu thông số model
  * `-result_forma`: Chọn loại tệp lưu thông số model (`csv`, `json`)

**Ví dụ 1: Thay đổi chiến lược xử lý dữ liệu**

```bash
python main.py --outlier_strategy drop --detection_params 3.0
```

**Ví dụ 2: Dự đoán nhanh cho một bệnh nhân khác**

```bash
python main.py --patient_info "60, 'Male', 'Cleveland', 'asymptomatic', 140.0, 260.0, False, 'normal', 150.0, True, 2.0, 'flat', 1.0, 'normal'"
```

-----

## 6. Dataset

Dự án sử dụng bộ dữ liệu **Heart Disease UCI**.

  * **Đường dẫn:** `Data/heart_disease_uci.csv`
  * **Biến mục tiêu (Target):** `num` (Dự đoán bệnh tim).

### Chi tiết các thuộc tính (Features)

| Tên cột | Mô tả chi tiết | Giá trị / Đơn vị |
| :--- | :--- | :--- |
| **id** | Mã định danh duy nhất của bệnh nhân | (Thường được loại bỏ khi huấn luyện) |
| **age** | Tuổi của bệnh nhân | Năm |
| **dataset** | Nơi thu thập dữ liệu (Nơi nghiên cứu) | Cleveland, Hungary, Switzerland, VA Long Beach |
| **sex** | Giới tính | `Male` (Nam), `Female` (Nữ) |
| **cp** | Loại đau ngực (Chest Pain Type) | `typical angina` (điển hình), `atypical angina` (không điển hình), `non-anginal` (không đau thắt), `asymptomatic` (không triệu chứng) |
| **trestbps**| Huyết áp lúc nghỉ | mm Hg (khi nhập viện) |
| **chol** | Mức Cholesterol huyết thanh | mg/dl |
| **fbs** | Đường huyết lúc đói \> 120 mg/dl | `True` (Có), `False` (Không) |
| **restecg** | Kết quả điện tâm đồ lúc nghỉ | `normal` (bình thường), `st-t abnormality`, `lv hypertrophy` |
| **thalach** | Nhịp tim tối đa đạt được | Nhịp/phút |
| **exang** | Đau thắt ngực do gắng sức (Exercise-induced angina) | `True` (Có), `False` (Không) |
| **oldpeak** | Độ chênh ST gây ra bởi vận động so với lúc nghỉ | Giá trị số (Numeric) |
| **slope** | Độ dốc của đoạn ST đỉnh vận động | `downsloping`, `flat`, `upsloping` |
| **ca** | Số lượng mạch chính được tô màu bởi fluoroscopy | 0 đến 3 |
| **thal** | Tình trạng Thalassemia | `normal`, `fixed defect`, `reversible defect` |
| **num** | Thuộc tính dự đoán (Mức độ bệnh) | Biến mục tiêu (Target attribute) |

