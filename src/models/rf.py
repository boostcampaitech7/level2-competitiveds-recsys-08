import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
import wandb
from src.utils import load_config


def rf_cv(X_train, y_train, n_splits=5, random_seed=42):
    # 5-fold 교차 검증 준비
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    # 각 폴드의 예측과 실제 값을 저장할 리스트
    oof_predictions = np.zeros(len(y_train))
    oof_targets = np.zeros(len(y_train))

    # 각 폴드의 모델을 저장할 리스트
    models = []

    config = load_config("configs/train_config.yaml")
    rf_params = config["rf"]  # catboost 파라미터 불러오기

    # 5-fold 교차 검증 수행
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
        # 각 폴드마다 새로운 wandb run 시작
        run = wandb.init(
            project="rf CV",
            name=f"rf_cv_fold_{fold}",
            reinit=True,
        )

        # wandb에 파라미터 로깅
        wandb.config.update(rf_params)

        print(f"Fold {fold}")

        # Train/Validation 데이터셋 분리
        X_fold, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # Random Forest 모델 생성
        rf_model = RandomForestRegressor(**rf_params)

        # 모델 학습
        rf_model.fit(X_fold, y_fold)

        # 검증 세트에 대한 예측
        oof_predictions[val_idx] = rf_model.predict(X_val)
        oof_targets[val_idx] = y_val

        # 모델 저장
        models.append(rf_model)

        # 폴드별 성능 로깅
        fold_mae = mean_absolute_error(y_val, oof_predictions[val_idx])
        fold_rmse = np.sqrt(mean_squared_error(y_val, oof_predictions[val_idx]))
        wandb.log({"fold-MAE": fold_mae, "fold-RMSE": fold_rmse})

        # wandb run 종료
        run.finish()

    # 최종 결과를 위한 새로운 wandb run 시작
    final_run = wandb.init(
        project="rf CV",
        name="rf_cv_final_results",
        reinit=True,
    )

    # 전체 OOF 성능 계산
    oof_mae = mean_absolute_error(oof_targets, oof_predictions)
    oof_rmse = np.sqrt(mean_squared_error(oof_targets, oof_predictions))
    oof_r2 = r2_score(oof_targets, oof_predictions)

    # wandb에 OOF 성능 로깅
    wandb.log({"oof-MAE": oof_mae, "oof-RMSE": oof_rmse, "oof-R²": oof_r2})

    # 결과 출력
    print("5-fold 교차 검증 RF 성능 (OOF):")
    print(f"MAE: {oof_mae:.2f}")
    print(f"RMSE: {oof_rmse:.2f}")
    print(f"R²: {oof_r2:.2f}")

    # 최종 wandb run 종료
    final_run.finish()

    return models
