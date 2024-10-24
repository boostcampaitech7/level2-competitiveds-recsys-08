import wandb
import numpy as np
import pickle
import yaml


# 플롯 제목 통일을 위해 LGBM 커스텀 콜백 설정
def lgb_wandb_callback():
    def callback(env):
        for data_name, metric_name, value, _ in env.evaluation_result_list:
            # 메트릭 이름 변경
            if metric_name == "l1":
                metric_name = "MAE"
            elif metric_name == "rmse":
                metric_name = "RMSE"

            # 로그 이름 생성
            log_name = f"{data_name}-{metric_name}"

            # wandb에 로그 기록
            wandb.log({log_name: value}, step=env.iteration)

    return callback


def save_model_to_pkl(models, name):
    # 피클로 저장
    lgb_save_path = "saved/models/"
    # Save each model using pickle
    for i, model in enumerate(models):
        model_filename = f"{lgb_save_path}{name}_model_fold_{i+1}.pkl"
        with open(model_filename, "wb") as file:
            pickle.dump(model, file)
        print(f"Model for fold {i+1} saved to {model_filename}")


def load_model_from_pkl(model, model_save_path):
    models = []
    for i in range(5):
        model_filename = f"{model_save_path}{model}_model_fold_{i+1}.pkl"
        with open(model_filename, "rb") as file:
            model = pickle.load(file)
            models.append(model)
            print(f"Model for fold {i+1} loaded from {model_filename}")

    return models


def load_config(path):
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config
