import lightgbm as lgb
from catboost import CatBoostRegressor

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
import wandb
from func.utils import lgb_wandb_callback



def train_lgb_with_wandb(X_train, y_train, lgb_params, X_holdout, y_holdout, project_name, experiment_name, entity_name):
    """

    """
    # wandb 초기화
    wandb.init(project=project_name, name=experiment_name, entity=entity_name)
    
    # wandb에 파라미터 로깅
    wandb.config.update(lgb_params)

    # LightGBM 데이터셋 생성
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_holdout, label=y_holdout, reference=train_data)

    # LightGBM 모델 학습
    lgb_model = lgb.train(
        lgb_params,
        train_data,
        valid_sets=[valid_data],
        valid_names='validation',
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=100),
            lgb_wandb_callback()
        ]
    )

    # Holdout 데이터 예측
    lgb_holdout_pred = lgb_model.predict(X_holdout, num_iteration=lgb_model.best_iteration)

    # 성능 메트릭 계산
    lgb_holdout_mae = mean_absolute_error(y_holdout, lgb_holdout_pred)
    lgb_holdout_rmse = np.sqrt(mean_squared_error(y_holdout, lgb_holdout_pred))
    lgb_holdout_r2 = r2_score(y_holdout, lgb_holdout_pred)

    # wandb에 성능 지표 로깅
    wandb.log({
        "holdout_mae": lgb_holdout_mae,
        "holdout_rmse": lgb_holdout_rmse,
        "holdout_r2": lgb_holdout_r2
    })

    print("Holdout 데이터셋 LGBM 성능:")
    print(f"MAE: {lgb_holdout_mae:.2f}")
    print(f"RMSE: {lgb_holdout_rmse:.2f}")
    print(f"R²: {lgb_holdout_r2:.2f}")

    wandb.finish()

    return lgb_model



def train_lgb(X_train, y_train, lgb_params, 
                          X_holdout=None, y_holdout=None, 
                          project_name=None, experiment_name=None, entity_name=None, 
                          X_test=None, sample_submission = None):
    """
        holdout데이터로 확인할 때 사용되는 경우와,
        전체 데이터 재학습하고 테스트 데이터 예측하는 거 합친 함수... -> 매우 비효율적일 것 같음
    """
    # LightGBM 데이터셋 생성
    train_data = lgb.Dataset(X_train, label=y_train)
    
    # wandb 초기화 및 파라미터 로깅
    if project_name and experiment_name and entity_name and X_holdout and y_holdout is not None:
        wandb.init(project=project_name, name=experiment_name, entity=entity_name)
        wandb.config.update(lgb_params)
        
        valid_data = lgb.Dataset(X_holdout, label=y_holdout, reference=train_data)

        # LightGBM 모델 학습
        lgb_model = lgb.train(
            lgb_params,
            train_data,
            valid_sets=[valid_data],
            valid_names='validation',
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(period=100),
                lgb_wandb_callback()
            ]
        )

        # Holdout 데이터 예측
        lgb_holdout_pred = lgb_model.predict(X_holdout, num_iteration=lgb_model.best_iteration)

        # 성능 메트릭 계산
        lgb_holdout_mae = mean_absolute_error(y_holdout, lgb_holdout_pred)
        lgb_holdout_rmse = np.sqrt(mean_squared_error(y_holdout, lgb_holdout_pred))
        lgb_holdout_r2 = r2_score(y_holdout, lgb_holdout_pred)

        # wandb에 성능 지표 로깅
        wandb.log({
            "holdout_mae": lgb_holdout_mae,
            "holdout_rmse": lgb_holdout_rmse,
            "holdout_r2": lgb_holdout_r2
        })

        print("Holdout 데이터셋 LGBM 성능:")
        print(f"MAE: {lgb_holdout_mae:.2f}")
        print(f"RMSE: {lgb_holdout_rmse:.2f}")
        print(f"R²: {lgb_holdout_r2:.2f}")

        wandb.finish()

        return lgb_model
    
    elif X_test and sample_submission is not None:
        lgb_model = lgb.train(
            lgb_params,
            train_data,
        )

        lgb_test_pred = lgb_model.predict(X_test)
        sample_submission['deposit'] = lgb_test_pred

        sample_submission.to_csv('output.csv', index=False, encoding='utf-8-sig')

        return 0
    
    return 0
    

def cross_validation_lgb_with_wandb(X_all, y_all, lgb_params, project_name, experiment_name, entity_name, n_splits=5, random_seed=42):
    # 5-fold 교차 검증 준비
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    # 각 폴드의 예측과 실제 값을 저장할 리스트
    oof_predictions = np.zeros(len(y_all))
    oof_targets = np.zeros(len(y_all))

    # 각 폴드의 모델을 저장할 리스트
    models = []

    # 5-fold 교차 검증 수행
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_all), 1):
        # 각 폴드마다 새로운 wandb run 시작
        run = wandb.init(project=project_name, name=f"lgb_cv_fold_{fold}", entity=entity_name, reinit=True)

        # wandb에 파라미터 로깅
        wandb.config.update(lgb_params)

        print(f"Fold {fold}")

        # Train/Validation 데이터셋 분리
        X_train, X_val = X_all.iloc[train_idx], X_all.iloc[val_idx]
        y_train, y_val = y_all.iloc[train_idx], y_all.iloc[val_idx]

        # LightGBM 데이터셋 생성
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # LightGBM 모델 학습
        model = lgb.train(
            lgb_params,
            train_data,
            valid_sets=[val_data],
            valid_names='validation',
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(period=100),
                lgb_wandb_callback()
            ]
        )

        # 검증 세트에 대한 예측
        oof_predictions[val_idx] = model.predict(X_val)
        oof_targets[val_idx] = y_val

        # 모델 저장
        models.append(model)

        # 폴드별 성능 로깅
        fold_mae = mean_absolute_error(y_val, oof_predictions[val_idx])
        fold_rmse = np.sqrt(mean_squared_error(y_val, oof_predictions[val_idx]))
        wandb.log({"fold-MAE": fold_mae, "fold-RMSE": fold_rmse})

        # wandb run 종료
        run.finish()

    # 최종 OOF 성능 계산 및 로깅
    final_run = wandb.init(project=project_name, name=experiment_name, entity=entity_name, reinit=True)

    # 전체 OOF 성능 계산
    oof_mae = mean_absolute_error(oof_targets, oof_predictions)
    oof_rmse = np.sqrt(mean_squared_error(oof_targets, oof_predictions))
    oof_r2 = r2_score(oof_targets, oof_predictions)

    # wandb에 OOF 성능 로깅
    wandb.log({
        "oof-MAE": oof_mae,
        "oof-RMSE": oof_rmse,
        "oof-R²": oof_r2
    })

    # 결과 출력
    print("5-fold 교차 검증 LGBM 성능 (OOF):")
    print(f"MAE: {oof_mae:.2f}")
    print(f"RMSE: {oof_rmse:.2f}")
    print(f"R²: {oof_r2:.2f}")

    # 특성 중요도 계산 (gain)
    feature_importance = np.mean([model.feature_importance(importance_type='gain') for model in models], axis=0)
    feature_names = X_all.columns.tolist()

    # 특성 중요도를 wandb에 로깅
    feature_importance_data = [
        [feature, importance] for feature, importance in zip(feature_names, feature_importance)
    ]
    feature_importance_table = wandb.Table(data=feature_importance_data, columns=["feature", "importance"])
    wandb.log({"feature_importance": wandb.plot.bar(feature_importance_table, "feature", "importance", title="Feature Importance (Gain)")})

    # 최종 wandb run 종료
    final_run.finish()


    return models, oof_predictions


def soft_voting_predict(models, X_test):
    predictions = np.column_stack([model.predict(X_test) for model in models])
    return np.mean(predictions, axis=1)
