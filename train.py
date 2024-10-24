import yaml
import sys
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold

import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
import wandb
from wandb.integration.lightgbm import log_summary
from wandb.integration.catboost import WandbCallback

import warnings
warnings.filterwarnings("ignore")

import src.data.preprocessor as pre
from src.models.lgbm import lgb_cv
from src.models.save_model import save_model_to_pkl
from src.utils import lgb_wandb_callback, load_config

def main():
    wandb.login()

    # config.yaml 로드
    config = load_config("configs/train_config.yaml")
    print(f"Project Name: {config['common']['project_name']}")

    # 랜덤시드 고정
    RANDOM_SEED = config['common']['random_seed']
    np.random.seed(RANDOM_SEED)

    # 데이터 경로와 전처리
    path = config['common']['data_path']
    data_preprocessor = pre.DataPreprocessor(path, "train")
    X_train, y_train = data_preprocessor.preprocess()
    
    # 모델 선택에 따른 학습
    if sys.argv[1] == "lgb":
        # lgb 학습
        lgb_params = config['lightgbm']  # LightGBM 파라미터 불러오기
        lgb_models = lgb_cv(X_train, y_train)
        lgb_models = lgb_cv(X_train, y_train, params = lgb_params, n_splits=config['common']['n_splits'], random_seed=RANDOM_SEED)
        save_model_to_pkl(lgb_models, sys.argv[1])

        return

    elif sys.argv[1] == "cat": 
        # catboost 학습 (추가구현 필요)
        cat_params = config['catboost']  # catboost 파라미터 불러오기
        cat_model = CatBoostRegressor(**cat_params)
        cat_model.fit(X_train, y_train) # catboosts에 cv 사용하려면 arg 추가
        save_model_to_pkl(cat_model, sys.argv[1])
    
    elif sys.argv[1] == "xgb":  # XGBoost (임시)
        pass
    
    else: # 앙상블 모델을 위한 추가 구현 (임시)
        return



if __name__ == "__main__":
    main()
