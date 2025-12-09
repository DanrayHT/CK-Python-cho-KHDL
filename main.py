import configparser
import argparse
import ast
from Script.DP import DataPreprocessor
from Script.LR import LogisticRegressionModel
from Script.SVM import SVMModel
from Script.RF import RandomForestModel
from Script.XGB import XGBoostModel
from Script.ModelSelector import ModelSelector
import sys
import os
import logging
import warnings

####################################################################################################################

warnings.filterwarnings('ignore')
# Tắt logging
logging.disable(logging.CRITICAL)

processed_df = None
preprocessor = None

####################################################################################################################

def load_config():
    """Đọc cấu hình từ file .ini được chỉ định và xử lý đối số dòng lệnh."""
    
    # Thiết lập Argparse (Đọc trước để lấy đường dẫn file config)
    parser = argparse.ArgumentParser(description="Model Training and Prediction Script.")
    
    # THÊM DÒNG MỚI: Đối số cho đường dẫn file cấu hình
    parser.add_argument('--config', type=str, default='configs/default.ini',
                        help='Đường dẫn tới file cấu hình (mặc định: configs/default.ini)')

    # Phân tích đối số chỉ để lấy đường dẫn config
    temp_args, _ = parser.parse_known_args()
    config_path = temp_args.config
    
    config = configparser.ConfigParser()
    
    # Kiểm tra nếu file config tồn tại
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"File cấu hình không tìm thấy tại đường dẫn: {config_path}. Hãy đảm bảo file tồn tại.")

    config.read(config_path)
    print(f"Đã đọc cấu hình từ file: {config_path}")
    
    # --- Tiếp tục thêm các đối số có thể ghi đè khác ---
    # Sử dụng parser chính để thêm các đối số khác và sau đó phân tích tất cả
    parser.add_argument('--input_file', type=str, 
                        default=config.get('PREPROCESSING', 'input_file', fallback='Data/heart_disease_uci.csv'), 
                        help="Đường dẫn tới file dữ liệu đầu vào.")
    
    parser.add_argument('--random_state', type=int, 
                        default=config.getint('PREPROCESSING', 'random_state', fallback=42), 
                        help="Seed cho quá trình ngẫu nhiên.")
                        
    parser.add_argument('--missing_strategy_num', type=str, 
                        default=config.get('PREPROCESSING', 'missing_strategy_num', fallback='median'), 
                        choices=['mean', 'median', 'most_frequent'],
                        help="Chiến lược điền giá trị thiếu cho cột số.")
                        
    parser.add_argument('--missing_strategy_cat', type=str, 
                        default=config.get('PREPROCESSING', 'missing_strategy_cat', fallback='most_frequent'), 
                        choices=['most_frequent', 'constant'],
                        help="Chiến lược điền giá trị thiếu cho cột phân loại.")
                        
    parser.add_argument('--outlier_strategy', type=str, 
                        default=config.get('PREPROCESSING', 'outlier_strategy', fallback='clip'), 
                        choices=['drop', 'clip', 'impute'],
                        help="Chiến lược xử lý outlier ('drop' hoặc 'clip').")
                        
    parser.add_argument('--detection_params', type=float, 
                        default=config.getfloat('PREPROCESSING', 'detection_params_z_score_threshold', fallback=2.5),
                        help="Ngưỡng Z-score cho phát hiện outlier.")
                        
    parser.add_argument('--patient_info', type=str, 
                        default=config.get('PATIENT_INFO', 'patient_data'),
                        help="Thông tin bệnh nhân mới dưới dạng chuỗi các giá trị được phân cách bằng dấu phẩy.")

    args = parser.parse_args()
    
    try:
        patient_info_list = list(ast.literal_eval(args.patient_info))
    except Exception as e:
        raise ValueError(f"Lỗi khi phân tích patient_info: {args.patient_info}. Đảm bảo định dạng chính xác. Lỗi: {e}")

    preprocessing_params = {
        'input_file': args.input_file,
        'random_state': args.random_state,
        'missing_strategy_num': args.missing_strategy_num,
        'missing_strategy_cat': args.missing_strategy_cat,
        'outlier_strategy': args.outlier_strategy,
        'detection_params': {'z_score_threshold': args.detection_params}
    }
    
    return preprocessing_params, patient_info_list

####################################################################################################################

if __name__ == "__main__":
    original_stdout = sys.stdout
    try:
        # --- BƯỚC 1: ĐỌC CẤU HÌNH VÀ THAM SỐ DÒNG LỆNH ---
        preprocessing_params, patient_info = load_config()

        input_file = preprocessing_params['input_file']
        
        # Tắt stdout tạm thời
        sys.stdout = open(os.devnull, 'w')
        
        try:
            # Khởi tạo và xử lý dữ liệu với các tham số đã đọc
            preprocessor_obj = DataPreprocessor(
                target_column='num', 
                random_state=preprocessing_params['random_state']
            )
            preprocessor_obj.load_data(input_file)
            
            # --- Thực thi Pipeline Tiền xử lý Tự động với các tham số đã đọc ---
            processed_data = preprocessor_obj.auto_process_by_data_types(
                missing_strategy_num=preprocessing_params['missing_strategy_num'], 
                missing_strategy_cat=preprocessing_params['missing_strategy_cat'],
                outlier_strategy=preprocessing_params['outlier_strategy'],
                detection_params=preprocessing_params['detection_params']
            )
            preprocessor_obj.save_processed_data('heart_disease_processed.csv')

            globals()['processed_df'] = processed_data
            globals()['preprocessor'] = preprocessor_obj

####################################################################################################################

            # Tách X và y
            X = processed_df.drop(columns=["num"])
            y_bin = (processed_df["num"] > 0).astype(int)
            rs = preprocessing_params['random_state']

            # Danh sách các model
            models = [
                LogisticRegressionModel(X=X, y=y_bin, C=1.0, penalty="l2", max_iter=1000, random_state=rs),
                SVMModel(X=X, y=y_bin, C=1.0, kernel="rbf", probability=True, random_state=rs),
                RandomForestModel(X=X, y=y_bin, n_estimators=1000, max_depth=None, random_state=rs),
                XGBoostModel(X=X, y=y_bin, n_estimators=2000, learning_rate=0.01, max_depth=6, random_state=rs)
            ]
            # tạo folder save
            save_folder = "models_all/"
            # Train và evaluate models
            for model in models:
                model.fit()
                
                path = f"{save_folder}{model.get_name()}.pkl"
                model.save_model(path)
                
                metric = model.evaluate()
                cv_score = model.cross_validate(cv=5)

            # Chọn best model
            test = ModelSelector(models=models, metric='recall')
            test.set_best_model()
            best_model = test.get_best_model()[0]["model"]

####################################################################################################################

            # Đọc thông tin bệnh nhân mới từ biến patient_info đã được load
            # patient_info đã được load từ file .ini hoặc từ đối số dòng lệnh
            
            # Preprocess bệnh nhân mới
            patient_info_df = preprocessor.preprocess_new_patient(
                new_patient_data=patient_info,
                processed_df=processed_data
            )

            # Dự đoán
            prediction = best_model.predict(patient_info_df)[0]
            proba = best_model.predict_proba(patient_info_df)[0]
            
        finally:
            # Khôi phục stdout
            sys.stdout.close()
            sys.stdout = original_stdout

        # Hiển thị kết quả (CHỈ PHẦN NÀY ĐƯỢC HIỂN THỊ)
        labels = {0: "Không bệnh", 1: "Có bệnh"}
        
        print(f"Dự đoán: **{labels[prediction]}**")
        print(f"Xác suất dự đoán: Không bệnh = {proba[0]:.2%}, Có bệnh = {proba[1]:.2%}")

    except Exception as e:
        if sys.stdout != original_stdout:
            sys.stdout.close()
            sys.stdout = original_stdout
        print(f"\nLỗi: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
