import sys
import pandas as pd
import numpy as np

# import wandb
import src.data.preprocessor as pre
from models.lgbm import lgb_cv
from src.models.catboost import cat_cv
from src.models.rf import rf_cv
from src.utils import load_config, save_model_to_pkl
from src.arg_parser import parse_args
import warnings

warnings.filterwarnings("ignore")

"""
This project uses Weights & Biases (wandb) for experiment tracking and visualization.
To enable wandb, uncomment the wandb-related code sections and install wandb:
    pip install wandb
"""


def main():
    # wandb.login()  # WandB 사용시 활성화

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

    # 모든 인자가 False인지 확인
    if not any(vars(args).values()):
        print("오류: 최소한 하나의 모델 옵션을 선택해야 합니다.")
        print("사용법: python train.py [-lgb] [-cat] [-rf]")
        sys.exit(1)

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

    if args.cat:
        # catboost 학습
        cat_models = cat_cv(
            X_train,
            y_train,
            n_splits=config["common"]["n_splits"],
            random_seed=RANDOM_SEED,
        )
        save_model_to_pkl(cat_models, "cat")

    if args.rf:
        # random forest 학습
        rf_models = rf_cv(
            X_train,
            y_train,
            n_splits=config["common"]["n_splits"],
            random_seed=RANDOM_SEED,
        )
        save_model_to_pkl(rf_models, "rf")


if __name__ == "__main__":
    main()
