import sys
from typing import Tuple, List

import pandas as pd

import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

from sklearn.metrics import root_mean_squared_error

def get_version():
    return f"""
        python: {sys.version}
        sklearn: {sklearn.__version__}
        pandas: {pd.__version__}
    """

def read_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df

def add_duration(df: pd.DataFrame, columns_dt: List[str] = ['tpep_dropoff_datetime', 'tpep_pickup_datetime']) -> pd.DataFrame:
    df[columns_dt[0]] = pd.to_datetime(df[columns_dt[0]])
    df[columns_dt[1]] = pd.to_datetime(df[columns_dt[1]])

    df['duration'] = df[columns_dt[0]] - df[columns_dt[1]]
    df['duration'] = df.duration.dt.total_seconds() / 60

    return df

def vectorize_data(df: pd.DataFrame, numerical: list[str] = ['duration'], categorical: list[str] = ['PULocationID', 'DOLocationID'], label: str = 'duration', dv: DictVectorizer = None) -> Tuple[Tuple[pd.DataFrame, pd.DataFrame], DictVectorizer]:
    df_features = df[categorical].astype(str)

    df_features = df_features[categorical].to_dict(orient='records')

    if dv is not None:
        print(f"Using existing DictVectorizer")
        df_features
        X_train = dv.transform(df_features)
    else:
        print(f"Creating new DictVectorizer")
        dv = DictVectorizer()
        X_train = dv.fit_transform(df_features)
    y_train = df[label]

    return ((X_train, y_train), dv)

def train_model(x_train: pd.DataFrame, y_train: pd.DataFrame, model: object = LinearRegression()) -> (LinearRegression, float):
    model.fit(x_train, y_train)
    y_predict = model.predict(x_train)
    rmse = root_mean_squared_error(y_train, y_predict)
    return model, rmse
