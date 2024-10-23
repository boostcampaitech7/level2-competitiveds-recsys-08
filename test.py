import pandas as pd
import numpy as np

import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
import pickle

import src.data.preprocessor as pre


def main():
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)

    path = "../data/"
    data_preprocessor = pre.DataPreprocessor(path, "test")
    X_test = data_preprocessor.preprocess()
    print(X_test)

    """
    피클 불러오기
    앙상블 파일 불러오기
    앙상블 진행
    제출파일 만들기
    """

    # 피클 불러오기
    model_save_path = "saved/models/"
    lgb_models = []
    # Load each model using pickle
    for i in range(5):
        model_filename = f"{model_save_path}lgb_model_fold_{i+1}.pkl"
        with open(model_filename, "rb") as file:
            model = pickle.load(file)
            lgb_models.append(model)
            print(f"Model for fold {i+1} loaded from {model_filename}")
