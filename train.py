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
import optuna

optuna.logging.set_verbosity(optuna.logging.INFO)  # optuna log 설정

import warnings

warnings.filterwarnings("ignore")

import src.data.preprocessor as pre
from src.utils import lgb_wandb_callback
from src.models.lgbm import lgb_cv
from src.models.save_model import save_model_to_pkl
import sys

def main():
    wandb.login()
    print(wandb.api.default_entity)

    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)

    path = "../data/"
    data_preprocessor = pre.DataPreprocessor(path, "train")
    X_train, y_train = data_preprocessor.preprocess()
    

    if sys.argv[1] == "lgb":
        # lgb 학습
        lgb_models = lgb_cv(X_train, y_train)
        save_model_to_pkl(lgb_models, sys.argv[1])

        return
    
    elif sys.argv[1] == "cat":
        return
    
    else: # 모든 모델에 대한 학습 진행
        return



if __name__ == "__main__":
    main()
