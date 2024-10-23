import pandas as pd
import numpy as np

import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
import pickle

import src.data.features as ft
from src.data.preprocessing import split_X_y


def main():
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)

    path = "../../data/"  # 알잘딱 수정
    test_data = pd.read_csv(path + "test.csv")
    sample_submission = pd.read_csv(path + "sample_submission.csv")

    # 추가
    # interest = pd.read_csv(path + 'interestRate.csv')
    park = pd.read_csv(path + "parkInfo.csv")
    school = pd.read_csv(path + "schoolinfo.csv")
    subway = pd.read_csv(path + "subwayInfo.csv")

    # 각 아파트에 대해 가까운 지하철역 거리 추가
    test_data = ft.calculate_nearest_subway_distance(test_data, subway)

    # 각 아파트에 대해 가까운 학교 거리 추가
    test_data = ft.calculate_nearest_school_distance(test_data, school)

    radius_km = 3
    item_name = "park"

    # 유니크한 아파트 좌표로 공원 개수와 밀도 계산 후 결과를 원래 데이터에 매핑
    test_data = ft.map_park_density(test_data, park, radius_km, item_name)

    # 각 레벨에 대해 다른 거리 범위를 설정
    distance_kms = {
        "elementary": 1,  # 1km 이내
        "middle": 5,  # 3km 이내
        "high": 5,  # 5km 이내
    }

    test_data = ft.map_school_level_counts(test_data, school, distance_kms, n_jobs=8)

    # 전체 재학습 데이터
    X_test, y_test = split_X_y(test_data, "deposit")

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

    X_test = X_test[train_columns]

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
