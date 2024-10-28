import wandb
import numpy as np
import pickle
import yaml
from typing import List, Any, Callable
from lightgbm import Booster


def lgb_wandb_callback() -> Callable:
    """
    LightGBM 모델 학습 중 Weights & Biases에 로그를 기록하기 위한 콜백 함수를 생성합니다.

    Returns:
        Callable: LightGBM 콜백 함수
    """
    def callback(env: Any) -> None:
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


def save_model_to_pkl(models: List[Booster], name: str, model_save_path: str = 'saved/models/') -> None:
    """
    LightGBM 모델들을 pickle 형식으로 저장합니다.

    Args:
        models (List[Booster]): 저장할 모델 리스트
        name (str): 저장할 모델의 이름
        model_save_path (str): 모델이 저장된 경로

    Returns:
        None
    """
    for i, model in enumerate(models):
        model_filename = f"{model_save_path}{name}_model_fold_{i+1}.pkl"
        with open(model_filename, "wb") as file:
            pickle.dump(model, file)
        print(f"Model for fold {i+1} saved to {model_filename}")


def load_model_from_pkl(name: str, model_save_path: str) -> List[Booster]:
    """
    Pickle 형식으로 저장된 LightGBM 모델들을 불러옵니다.

    Args:
        name (str): 불러올 모델의 이름
        model_save_path (str): 모델이 저장된 경로

    Returns:
        List[Booster]: 불러온 LightGBM 모델 리스트
    """
    models = []
    for i in range(5):
        model_filename = f"{model_save_path}{name}_model_fold_{i+1}.pkl"
        with open(model_filename, "rb") as file:
            model = pickle.load(file)
            models.append(model)
            print(f"Model for fold {i+1} loaded from {model_filename}")

    return models


def load_config(path: str) -> dict:
    """
    YAML 설정 파일을 불러옵니다.

    Args:
        path (str): YAML 파일의 경로

    Returns:
        dict: 불러온 설정 정보를 담은 딕셔너리
    """
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config
