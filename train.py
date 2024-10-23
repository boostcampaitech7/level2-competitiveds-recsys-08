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
import pickle


def main():
    wandb.login()
    print(wandb.api.default_entity)

    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)

    path = "../data/"
    data_preprocessor = pre.DataPreprocessor(path, "train")
    X_train, y_train = data_preprocessor.preprocess()

    # lgb 학습
    lgb_models = lgb_cv(X_train, y_train)
    # 피클로 저장
    lgb_save_path = "saved/models/"
    # Save each model using pickle
    for i, model in enumerate(lgb_models):
        model_filename = f"{lgb_save_path}lgb_model_fold_{i+1}.pkl"
        with open(model_filename, "wb") as file:
            pickle.dump(model, file)
        print(f"Model for fold {i+1} saved to {model_filename}")


main()
