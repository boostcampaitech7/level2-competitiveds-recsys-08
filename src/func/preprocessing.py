import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(data):
    
    processed_data = data.drop_duplicates(subset=data.columns.drop('index'), keep='first')

    return processed_data

def split_X_y(data, target):
    """
    데이터를 독립변수와 종속변수로 분리하는 함수.
    
    Args:
    - data (pd.DataFrame): 전체 데이터셋
    - target (str): 타겟 컬럼명 (예: 'deposit')
    
    Returns:
    - X_data (pd.DataFrame): 독립변수 df
    - y_data (pd.DataFrame): 종속변수 df
    """
    
    X_data = data.drop(columns=[target])
    y_data = data[target]
    
    return X_data, y_data

def split_holdout_data(data, holdout_start, holdout_end):
    """
    데이터를 학습 데이터와 Holdout 데이터로 분리하는 함수.
    
    Args:
    - data (pd.DataFrame): 전체 데이터셋
    - holdout_start (int): Holdout 데이터의 시작 날짜 (예: '2023-07-01')
    - holdout_end (int): Holdout 데이터의 종료 날짜 (예: '2023-12-31')
    
    Returns:
    - train_data (pd.DataFrame): 학습에 사용할 데이터
    - holdout_data (pd.DataFrame): Holdout에 사용할 데이터
    """
    
    holdout_data = data[(data['contract_year_month'] >= holdout_start) & (data['contract_year_month'] <= holdout_end)]
    train_data = data[~((data['contract_year_month'] >= holdout_start) & (data['contract_year_month'] <= holdout_end))]
    
    return train_data, holdout_data

