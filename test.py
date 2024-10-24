import pandas as pd
import numpy as np
import pickle
import yaml

import lightgbm as lgb
from catboost import CatBoostRegressor, Pool

import src.data.preprocessor as pre
from src.models.ensemble import vote_soft
from src.utils import *

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
    lgb_models = load_model_from_pkl('lgb', model_save_path)
    lgb_predictions = vote_soft(lgb_models, X_test)

    # catboost 모델 불러오기
    # cat_models = load_model_from_pkl('cat', model_save_path)
    # cat_predictions = vote_soft(cat_models, X_test)

    # 위 모델들 예측 결과 가지고 소프트 보팅?


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