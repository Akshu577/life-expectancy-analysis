"""The cleaning and modelling steps from Life_Expectancy_Analysis.ipynb, as functions the app can reuse."""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# NumPy on macOS Accelerate emits spurious matmul RuntimeWarnings; results are unaffected.
warnings.filterwarnings("ignore", message=".*encountered in matmul")

ROOT = Path(__file__).resolve().parents[1]
TARGET = "Life expectancy"
BASE_COLS = ["Adult Mortality", "infant deaths", "Alcohol", "percentage expenditure", "Hepatitis B",
             "Measles", "BMI", "under-five deaths", "Polio", "Total expenditure", "Diphtheria",
             "HIV/AIDS", "GDP", "Population", "thinness 1-19 years", "thinness 5-9 years",
             "Income composition of resources", "Schooling"]


def add_features(data):
    data["Population Size"] = pd.cut(data["Population"], bins=[0, 30000, 100000, np.inf],
                                     labels=["Small", "Medium", "Large"], right=False)
    data["Lifestyle"] = data["Alcohol"] * data["BMI"]
    data["Economy"] = data["GDP"] * data["Population"]
    data["Death Ratio"] = data["Adult Mortality"] / (data["infant deaths"] + 1)
    return data


def load_raw():
    df = pd.read_csv(ROOT / "Life_Expectancy_Data.csv")
    df.columns = df.columns.str.strip().str.replace(r"\s+", " ", regex=True)
    return df


def clean(df):
    """Impute, log-transform skewed columns, drop collinear features and cap outliers, as in the notebook."""
    df = add_features(df.copy())
    df = df.dropna(subset=[TARGET])
    df.loc[df["Schooling"] == 0, "Schooling"] = np.nan
    df.loc[df["Income composition of resources"] == 0, "Income composition of resources"] = np.nan
    for col in BASE_COLS:
        df[col] = df[col].fillna(df.groupby("Country")[col].transform("median"))
        df[col] = df[col].fillna(df[col].median())
    df = add_features(df)
    raw = df.copy()

    predictors = BASE_COLS + ["Lifestyle", "Economy", "Death Ratio"]
    skew = df[predictors].skew()
    log_cols = skew[skew > 2].index.tolist()
    for col in log_cols:
        df[col] = np.log1p(df[col])

    corr = df[predictors + [TARGET]].corr()
    target_corr = df[predictors].corrwith(df[TARGET]).abs()
    to_drop = []
    for i in range(len(predictors)):
        for j in range(i + 1, len(predictors)):
            a, b = predictors[i], predictors[j]
            if a in to_drop or b in to_drop:
                continue
            if abs(corr.loc[a, b]) > 0.75:
                to_drop.append(a if target_corr[a] < target_corr[b] else b)
    features = [c for c in predictors if c not in to_drop]

    q1, q3 = df[features].quantile(0.25), df[features].quantile(0.75)
    lower, upper = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    for col in features:
        df[col] = df[col].clip(lower[col], upper[col])
    return df, raw, features, log_cols, lower, upper


def fit_small_model(df, features, n=4):
    """Forward selection on the notebook's train/test split; returns a model on the first n features."""
    X = df[features].copy()
    X["Developed"] = (df["Status"] == "Developed").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, df[TARGET], test_size=0.25, random_state=42)
    selected = []
    while len(selected) < n:
        best = min((f for f in X.columns if f not in selected),
                   key=lambda f: np.mean((y_test - LinearRegression().fit(X_train[selected + [f]], y_train)
                                          .predict(X_test[selected + [f]])) ** 2))
        selected.append(best)
    model = LinearRegression().fit(X_train[selected], y_train)
    rmse = float(np.sqrt(np.mean((y_test - model.predict(X_test[selected])) ** 2)))
    return model, selected, rmse
