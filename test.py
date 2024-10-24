import pandas as pd
import numpy as np
import pickle
import yaml

import lightgbm as lgb
from catboost import CatBoostRegressor, Pool

import src.data.preprocessor as pre

def load_config(path): # train.py와 중복
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config

def main():

    # config 파일 불러오기
    config = load_config("configs/test_config.yaml")

    # 공통 설정
    RANDOM_SEED = config['common']['random_seed']
    np.random.seed(RANDOM_SEED)

    # 테스트 데이터 전처리
    path = config['common']['data_path']
    data_preprocessor = pre.DataPreprocessor(path, "test")
    X_test = data_preprocessor.preprocess()
    print("Test data preprocessed:", X_test.shape)

    # 피클 불러오기 경로
    model_save_path = config['common']['model_save_path']
    
    # LightGBM 모델 불러오기
    lgb_models = []
    for i in range(5):  # kfold 5개 모델 불러오기
        model_filename = f"{model_save_path}lgb_model_fold_{i+1}.pkl"
        with open(model_filename, "rb") as file:
            model = pickle.load(file)
            lgb_models.append(model)
            print(f"Model for fold {i+1} loaded from {model_filename}")

    # 예측 진행 (LightGBM 앙상블 예시)
    lgb_predictions = np.zeros(X_test.shape[0])  # 예측 결과 저장할 배열
    for model in lgb_models:
        lgb_predictions += model.predict(X_test) / len(lgb_models)  # 앙상블 평균 예측

    # 최종 예측 출력
    print("Final LightGBM predictions (ensemble):", lgb_predictions)

    # 제출 파일 만들기
    submission = pd.DataFrame({
        "Id": range(len(lgb_predictions)),  # Id 컬럼은 데이터에 맞게 수정 필요
        "Prediction": lgb_predictions
    })
    submission_filename = "submission.csv"
    submission.to_csv(submission_filename, index=False)
    print(f"Submission file saved as {submission_filename}")


if __name__ == "__main__":
    main()

    """
    피클 불러오기
    앙상블 파일 불러오기
    앙상블 진행
    제출파일 만들기
    """