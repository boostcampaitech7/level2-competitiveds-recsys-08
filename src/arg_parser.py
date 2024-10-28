import argparse


def parse_args():
    """
    명령줄 인자를 파싱하여 학습할 모델 옵션을 설정합니다.

    이 함수는 다음과 같은 명령줄 옵션을 지원합니다:
    -lgb: LightGBM 모델 학습
    -cat: CatBoost 모델 학습
    -rf: Random Forest 모델 학습

    Returns:
        argparse.Namespace: 파싱된 명령줄 인자를 포함하는 Namespace 객체

    Example:
        python script.py -lgb -cat
        이 명령은 LightGBM과 CatBoost 모델을 학습하도록 설정합니다.
    """
    parser = argparse.ArgumentParser(description="Train models with specified options.")
    parser.add_argument("-lgb", action="store_true", help="Train LightGBM model")
    parser.add_argument("-cat", action="store_true", help="Train CatBoost model")
    parser.add_argument("-rf", action="store_true", help="Train Random Forest model")

    return parser.parse_args()
