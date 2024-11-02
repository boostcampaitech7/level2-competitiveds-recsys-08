from catboost import CatBoostRegressor, Pool
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

# import wandb
from src.utils import load_config

"""
This project uses Weights & Biases (wandb) for experiment tracking and visualization.
To enable wandb, uncomment the wandb-related code sections and install wandb:
    pip install wandb
"""


def cat_cv(X_train, y_train, n_splits=5, random_seed=42):
    # 5-fold 교차 검증 준비
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    # 각 폴드의 예측과 실제 값을 저장할 리스트
    oof_predictions = np.zeros(len(y_train))
    oof_targets = np.zeros(len(y_train))

    # 각 폴드의 모델을 저장할 리스트
    models = []

    config = load_config("configs/train_config.yaml")
    cat_params = config["catboost"]  # catboost 파라미터 불러오기

    # 5-fold 교차 검증 수행
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
        # # 각 폴드마다 새로운 wandb run 시작
        # run = wandb.init(
        #     project="cat CV",
        #     name=f"cat_cv_fold_{fold}",
        #     reinit=True,
        # )

        # # wandb에 파라미터 로깅
        # wandb.config.update(cat_params)

        print(f"Fold {fold}")

        # Train/Validation 데이터셋 분리
        X_fold, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # CatBoost 데이터셋 생성
        train_pool = Pool(X_fold, y_fold)
        valid_pool = Pool(X_val, y_val)

        # CatBoost 모델 생성
        cat_model = CatBoostRegressor(**cat_params)

        # 모델 학습
        cat_model.fit(
            train_pool,
            eval_set=valid_pool,
            use_best_model=True,
        )

        # 검증 세트에 대한 예측
        oof_predictions[val_idx] = cat_model.predict(X_val)
        oof_targets[val_idx] = y_val

        # 모델 저장
        models.append(cat_model)

        # 폴드별 성능 로깅
        fold_mae = mean_absolute_error(y_val, oof_predictions[val_idx])
        fold_rmse = np.sqrt(mean_squared_error(y_val, oof_predictions[val_idx]))
        # wandb.log({"fold-MAE": fold_mae, "fold-RMSE": fold_rmse})

        # 결과 출력
        print(f"fold {fold} MAE: {fold_mae:.2f}, fold {fold} RMSE: {fold_rmse:.2f}")

        # # wandb run 종료
        # run.finish()

    # # 최종 결과를 위한 새로운 wandb run 시작
    # final_run = wandb.init(
    #     project="cat CV",
    #     name="cat_cv_final_results",
    #     reinit=True,
    # )

    # 전체 OOF 성능 계산
    oof_mae = mean_absolute_error(oof_targets, oof_predictions)
    oof_rmse = np.sqrt(mean_squared_error(oof_targets, oof_predictions))
    oof_r2 = r2_score(oof_targets, oof_predictions)

    # # wandb에 OOF 성능 로깅
    # wandb.log({"oof-MAE": oof_mae, "oof-RMSE": oof_rmse, "oof-R²": oof_r2})

    # 결과 출력
    print("5-fold 교차 검증 Cat 성능 (OOF):")
    print(f"MAE: {oof_mae:.2f}")
    print(f"RMSE: {oof_rmse:.2f}")
    print(f"R²: {oof_r2:.2f}")

    # # 특성 중요도 계산 (gain)
    # feature_importance = np.mean(
    #     [cat_model.get_feature_importance() for model in models], axis=0
    # )
    # feature_names = X_train.columns.tolist()

    # # 특성 중요도를 wandb에 로깅
    # feature_importance_data = [
    #     [feature, importance]
    #     for feature, importance in zip(feature_names, feature_importance)
    # ]
    # feature_importance_table = wandb.Table(
    #     data=feature_importance_data, columns=["feature", "importance"]
    # )
    # wandb.log(
    #     {
    #         "feature_importance": wandb.plot.bar(
    #             feature_importance_table,
    #             "feature",
    #             "importance",
    #             title="Feature Importance",
    #         )
    #     }
    # )

    # # 최종 wandb run 종료
    # final_run.finish()

    return models
