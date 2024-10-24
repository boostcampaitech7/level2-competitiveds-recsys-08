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
    RANDOM_SEED = config["common"]["random_seed"]
    np.random.seed(RANDOM_SEED)

    # 테스트 데이터 전처리
    path = config["common"]["data_path"]
    data_preprocessor = pre.DataPreprocessor(path, "test")
    X_test = data_preprocessor.preprocess()
    print("Test data preprocessed:", X_test.shape)

    # 피클 불러오기 경로
    model_save_path = config["common"]["model_save_path"]

    # LightGBM 모델 불러오기
    lgb_models = load_model_from_pkl("lgb", model_save_path)
    cat_models = load_model_from_pkl("cat", model_save_path)
    rf_models = load_model_from_pkl("rf", model_save_path)
    lgb_predictions = vote_soft(lgb_models, X_test)
    cat_predictions = vote_soft(cat_models, X_test)
    rf_predictions = vote_soft(rf_models, X_test)

    emsemble_predictions = (lgb_predictions + cat_predictions + rf_predictions) / 3

    # 제출 파일 만들기

    sample_submission = pd.read_csv(path + "sample_submission.csv")
    sample_submission["deposit"] = emsemble_predictions
    sample_submission.to_csv("output.csv", index=False, encoding="utf-8-sig")
    print("Submission file saved as 'output.csv'")


if __name__ == "__main__":
    main()

    """
    피클 불러오기
    앙상블 파일 불러오기
    앙상블 진행
    제출파일 만들기
    """
