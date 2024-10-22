import pandas as pd
import numpy as np

import wandb


import func.preprocessing as pp
import func.features as ft
import func.models as mdl
from func.utils import lgb_wandb_callback

def main():
    wandb.login()
    print(wandb.api.default_entity)
    
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)

    # csv 파일 불러오기
    path = "../../data/"  # 알잘딱 수정
    file_dict = {'train':'train_data', 'test':'test_data', 
             'sample_submission':'sample_submission', 'parkInfo':'park', 
             'schoolinfo':'school', 'subwayInfo':'subway'}
    for f, d in file_dict.items():
        exec(f"{d} = pp.load_data(path + '{f}.csv')")

    # 중복 값 처리
    train_data = pp.preprocess_data(train_data)

    # 각 아파트에 대해 가까운 지하철역 거리 추가
    train_data = ft.calculate_nearest_subway_distance(train_data, subway)
    test_data = ft.calculate_nearest_subway_distance(test_data, subway)

    # 각 아파트에 대해 가까운 학교 거리 추가
    train_data = ft.calculate_nearest_school_distance(train_data, school)
    test_data = ft.calculate_nearest_school_distance(test_data, school)
    
    # 공원 밀도 계산
    radius_km = 3
    item_name = 'park'
    train_data = ft.map_item_density_with_area(train_data, park, radius_km, item_name)
    test_data = ft.map_item_density_with_area(test_data, park, radius_km, item_name)

    # 특정 거리 내 레벨별 학교 개수
    distance_kms = {
        'elementary': 1, 
        'middle': 5,    
        'high': 5      
    }
    train_data = ft.map_school_level_counts(train_data, school, distance_kms, n_jobs=8)
    test_data = ft.map_school_level_counts(test_data, school, distance_kms, n_jobs=8)

    
    # 전체 재학습 데이터 설정
    all_data = train_data.copy()
    X_all, y_all = pp.split_X_y(all_data, 'deposit')
    X_test = test_data.copy()

    # train, holdout 데이터 설정
    holdout_start, holdout_end = 202307, 202312
    train_data, holdout_data = pp.split_holdout_data(train_data.copy(), holdout_start, holdout_end)
    X_train, y_train = pp.split_X_y(train_data, 'deposit')
    X_holdout, y_holdout = pp.split_X_y(holdout_data, 'deposit')


    train_columns = ['area_m2', 'contract_year_month', 'floor', 'built_year', 'latitude', 'longitude',
                 'nearest_subway_distance',  'nearest_elementary_distance', 'nearest_middle_distance',
                 'nearest_high_distance', 'park_density', 'elementary', 'middle', 'high']

    X_train = X_train[train_columns]
    X_holdout = X_holdout[train_columns]
    X_all = X_all[train_columns]
    X_test = X_test[train_columns]


    # 모델 부분 미완


if __name__ == '__main__':
    main()
