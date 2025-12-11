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
sys.stdout.reconfigure(encoding='utf-8')


####################################################################################################################

warnings.filterwarnings('ignore')

log_filename = 'model_training.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_filename, mode='w', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)
logger.info("=" * 80)
logger.info("PIPELINE HUẤN LUYỆN MÔ HÌNH BẮT ĐẦU")
logger.info("=" * 80)

processed_df = None
preprocessor = None

####################################################################################################################

def load_config():
    """Đọc cấu hình từ file .ini được chỉ định và xử lý đối số dòng lệnh."""
    
    # Thiết lập Argparse (Đọc trước để lấy đường dẫn file config)
    parser = argparse.ArgumentParser(description="Script Huấn Luyện Mô Hình và Dự Đoán.")
    
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
    
    parser.add_argument('--test_size', type=float, 
                        default=config.getfloat('MODEL', 'test_size', fallback=0.3),
                        help="Chọn kích cỡ tập test (từ 0->1)")
                        
    parser.add_argument('--patient_info', type=str, 
                        default=config.get('PATIENT_INFO', 'patient_data'),
                        help="Thông tin bệnh nhân mới dưới dạng chuỗi các giá trị được phân cách bằng dấu phẩy.")
    
    parser.add_argument('--metric', type=str,
                        default=config.get('MODEL_SELECTION', 'metric', fallback='recall'),
                        choices=['accuracy', 'f1', 'precision', 'recall'],
                        help="Chỉ số để lựa chọn mô hình tốt nhất (ModelSelector).")
    # Thêm tham số đường dẫn file kết quả
    parser.add_argument('--result_name', type=str, 
                        default=config.get('OUTPUT', 'result_name', fallback='models_data/experiment_results'), 
                        help="Đường dẫn để lưu file kết quả và tên file.")

    # Thêm tham số định dạng file (csv/json)
    parser.add_argument('--result_format', type=str, 
                        default=config.get('OUTPUT', 'format', fallback='csv'), 
                        help="Định dạng file lưu trữ (csv, json).")

    args = parser.parse_args()
    
    try:
        patient_info_list = list(ast.literal_eval(args.patient_info))
    except Exception as e:
        raise ValueError(f"Lỗi khi phân tích patient_info: {args.patient_info}. Đảm bảo định dạng chính xác. Lỗi: {e}")
    
    random_state_final = None if args.random_state == -1 else args.random_state

    preprocessing_params = {
        'input_file': args.input_file,
        'random_state': random_state_final,
        'missing_strategy_num': args.missing_strategy_num,
        'missing_strategy_cat': args.missing_strategy_cat,
        'outlier_strategy': args.outlier_strategy,
        'detection_params': {'z_score_threshold': args.detection_params}
    }

    model_info_params = {
        'test_size': args.test_size
    }

    model_selection_params = {
        'metric': args.metric
    }

    result_param = {
        'name': args.result_name,
        'type': args.result_format
    }
    
    return preprocessing_params, model_info_params, patient_info_list, model_selection_params, result_param

####################################################################################################################

if __name__ == "__main__":
    original_stdout = sys.stdout
    try:
        # --- BƯỚC 1: ĐỌC CẤU HÌNH VÀ THAM SỐ DÒNG LỆNH ---
        logger.info("BƯỚC 1: Đọc cấu hình và tham số dòng lệnh")
        preprocessing_params, model_info_params, patient_info, model_selection_params, result_param = load_config()

        input_file = preprocessing_params['input_file']
        logger.info(f"File đầu vào: {input_file}")
        logger.info(f"Chỉ số lựa chọn mô hình: {model_selection_params['metric']}")
        
        # Tắt stdout tạm thời
        sys.stdout = open(os.devnull, 'w')
        
        try:
            # Khởi tạo và xử lý dữ liệu với các tham số đã đọc
            logger.info("\nBƯỚC 2: Tiền xử lý dữ liệu")
            preprocessor_obj = DataPreprocessor(
                target_column='num', 
                random_state=preprocessing_params['random_state']
            )
            logger.info("Đang tải dữ liệu...")
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
            logger.info("\nBƯỚC 3: Chuẩn bị dữ liệu cho mô hình")
            X = processed_df.drop(columns=["num"])
            y_bin = (processed_df["num"] > 0).astype(int)
            logger.info(f"Kích thước đặc trưng: {X.shape}, Phân bố nhãn: {y_bin.value_counts().to_dict()}")
            rs = preprocessing_params['random_state']

            # Danh sách các model
            logger.info("\nBƯỚC 4: Huấn luyện và tối ưu hóa mô hình")
            models = [
                LogisticRegressionModel(X=X, y=y_bin, test_size=model_info_params['test_size'], C=1.0, penalty="l2", max_iter=1000, random_state=rs),
                SVMModel(X=X, y=y_bin, test_size=model_info_params['test_size'], C=1.0, kernel="rbf", probability=True, random_state=rs),
                RandomForestModel(X=X, y=y_bin, test_size=model_info_params['test_size'], n_estimators=1000, max_depth=None, random_state=rs),
                XGBoostModel(X=X, y=y_bin, test_size=model_info_params['test_size'], n_estimators=2000, learning_rate=0.01, max_depth=6, random_state=rs)
            ]
            param_grids = {
                "LogisticRegression": {"C": [0.01, 0.1, 1, 10], "penalty": ["l2"], "max_iter":[500,1000]},
                "SVM": {"C": [0.1, 1, 10], "gamma": ["scale","auto"], "kernel": ["rbf"]},
                "RandomForest": {"n_estimators": [50,100,200], "max_depth": [None,5,10]},
                "XGBoost": {"n_estimators": [50,100,200], "learning_rate": [0.01,0.1,0.2], "max_depth":[3,6,9]}
            }
            # tạo folder save
            save_folder = "models_all/"
            # Train và evaluate models
            logger.info(f"Đang huấn luyện {len(models)} mô hình...")
            for model in models:
                logger.info(f"\n{'='*60}")
                logger.info(f"Đang huấn luyện mô hình: {model.get_name()}")
                logger.info(f"{'='*60}")
                model.split_data()
                # Tối ưu siêu tham số nếu có param_grid
                if model.name in param_grids:
                    best = model.optimize_params(
                        param_grid=param_grids[model.name],
                        search="grid",
                        cv=3,
                        scoring=model_selection_params['metric'],
                        n_jobs=-1,
                        verbose=1
                    )
                model.fit()
                
                path = f"{save_folder}{model.get_name()}.pkl"
                model.save_model(path)
                
                metric = model.evaluate()
                cv_score = model.cross_validate(cv=5)
                
                # LƯU KẾT QUẢ THỰC NGHIỆM VÀO FILE CSV
                model.save_experiment_results(
                    filepath=result_param['name'] + '.' + result_param['type'],
                    format=result_param['type']
                )
                logger.info(f"Đã lưu kết quả thực nghiệm của {model.get_name()}")

            # Chọn best model
            logger.info("\nBƯỚC 5: Lựa chọn mô hình tốt nhất")
            test = ModelSelector(models=models, metric=model_selection_params['metric'])
            test.set_best_model()
            best_model = test.get_best_model()[0]["model"]
            logger.info(f"Mô hình tốt nhất đã chọn: {best_model.get_name()}")
            logger.info(f"{model_selection_params['metric']} tốt nhất: {test.get_best_model()[0]['score']:.4f}")

####################################################################################################################

            # Đọc thông tin bệnh nhân mới từ biến patient_info đã được load
            # patient_info đã được load từ file .ini hoặc từ đối số dòng lệnh
            
            # Preprocess bệnh nhân mới
            logger.info("\nBƯỚC 6: Dự đoán cho bệnh nhân mới")
            patient_info_df = preprocessor.preprocess_new_patient(
                new_patient_data=patient_info,
                processed_df=processed_data
            )

            # Dự đoán
            prediction = best_model.predict(patient_info_df)[0]
            proba = best_model.predict_proba(patient_info_df)[0]
            logger.info(f"Dự đoán: {prediction}, Xác suất: {proba}")
            
        finally:
            # Khôi phục stdout
            sys.stdout.close()
            sys.stdout = original_stdout

        # Hiển thị kết quả (CHỈ PHẦN NÀY ĐƯỢC HIỂN THỊ)
        labels = {0: "Không bệnh", 1: "Có bệnh"}
        
        logger.info("\n" + "="*80)
        logger.info("KẾT QUẢ CUỐI CÙNG")
        logger.info("="*80)
        print(f"Dự đoán: **{labels[prediction]}**")
        print(f"Xác suất dự đoán: Không bệnh = {proba[0]:.2%}, Có bệnh = {proba[1]:.2%}")
        logger.info(f"Log huấn luyện đã lưu tại: {log_filename}")

    except Exception as e:
        if sys.stdout != original_stdout:
            sys.stdout.close()
            sys.stdout = original_stdout
        print(f"\nLỗi: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
