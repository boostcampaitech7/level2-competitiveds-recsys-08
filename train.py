import sys
import pandas as pd
import numpy as np

import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
import wandb

import warnings

warnings.filterwarnings("ignore")

import src.data.preprocessor as pre
from src.models.lgbm import lgb_cv
from src.utils import load_config, save_model_to_pkl
from src.arg_parser import parse_args


def main():
    wandb.login()

    # cli option parser 로드
    args = parse_args()
    # config.yaml 로드
    config = load_config("configs/train_config.yaml")
    print(f"Project Name: {config['common']['project_name']}")

    # 랜덤시드 고정
    RANDOM_SEED = config["common"]["random_seed"]
    np.random.seed(RANDOM_SEED)

    # 데이터 경로와 전처리
    path = config["common"]["data_path"]
    data_preprocessor = pre.DataPreprocessor(path, "train")
    X_train, y_train = data_preprocessor.preprocess()

    # 모델 선택에 따른 학습
    if args.lgb:
        # lgb 학습
        lgb_models = lgb_cv(
            X_train,
            y_train,
            n_splits=config["common"]["n_splits"],
            random_seed=RANDOM_SEED,
        )
        save_model_to_pkl(lgb_models, "lgb")

        return

    elif args.cat:
        # catboost 학습 (추가구현 필요)
        cat_params = config["catboost"]  # catboost 파라미터 불러오기
        cat_model = CatBoostRegressor(**cat_params)
        cat_model.fit(X_train, y_train)  # catboosts에 cv 사용하려면 arg 추가
        save_model_to_pkl(cat_model, "cat")

    elif args.xgb:  # XGBoost (임시)
        pass

    else:
        print("No valid model option provided. Use -lgb, -cat, or -xgb.")


if __name__ == "__main__":
    main()
