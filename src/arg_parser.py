import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train models with specified options.")
    parser.add_argument("-lgb", action="store_true", help="Train LightGBM model")
    parser.add_argument("-cat", action="store_true", help="Train CatBoost model")
    parser.add_argument("-rf", action="store_true", help="Train Random Forest model")
    parser.add_argument("-xgb", action="store_true", help="Train XGBoost model")

    return parser.parse_args()
