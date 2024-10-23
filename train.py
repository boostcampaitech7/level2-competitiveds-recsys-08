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

import src.data.preprocessing as pp
import src.data.features as ft
import src.models.models as mdl
from src.utils import lgb_wandb_callback
from src.models.lgbm import lgb_cv
import pickle


def main():
    wandb.login()
    print(wandb.api.default_entity)

    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)

    path = "../../data/"  # 알잘딱 수정
    train_data = pd.read_csv(path + "train.csv")

    # 추가
    # interest = pd.read_csv(path + 'interestRate.csv')
    park = pd.read_csv(path + "parkInfo.csv")
    school = pd.read_csv(path + "schoolinfo.csv")
    subway = pd.read_csv(path + "subwayInfo.csv")

    train_data = pp.remove_duplicated_data(train_data)
    # 각 아파트에 대해 가까운 지하철역 거리 추가
    train_data = ft.calculate_nearest_subway_distance(train_data, subway)
    # 각 아파트에 대해 가까운 학교 거리 추가
    train_data = ft.calculate_nearest_school_distance(train_data, school)

    radius_km = 3
    item_name = "park"

    # 유니크한 아파트 좌표로 공원 개수와 밀도 계산 후 결과를 원래 데이터에 매핑
    train_data = ft.map_park_density(train_data, park, radius_km, item_name)

    # 각 레벨에 대해 다른 거리 범위를 설정
    distance_kms = {
        "elementary": 1,  # 1km 이내
        "middle": 5,  # 5km 이내
        "high": 5,  # 5km 이내
    }

    train_data = ft.map_school_level_counts(train_data, school, distance_kms, n_jobs=8)

    # 전체 재학습 데이터
    X_train, y_train = pp.split_X_y(train_data, "deposit")

    # 피처 선택
    train_columns = [
        "area_m2",
        "contract_year_month",
        "floor",
        "built_year",
        "latitude",
        "longitude",
        "nearest_subway_distance",
        "nearest_elementary_distance",
        "nearest_middle_distance",
        "nearest_high_distance",
        "park_density",
        "elementary",
        "middle",
        "high",
    ]

    X_train = X_train[train_columns]

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
