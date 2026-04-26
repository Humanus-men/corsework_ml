import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import load_and_prepare, regression_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
DATA_PATH = os.path.join(PROJECT_DIR, 'data', 'Данные_для_курсовои_Классическое_МО.xlsx')

df_orig, df, features = load_and_prepare(DATA_PATH)
X_train, X_test, y_train, y_test, selector, scaler = regression_split(df, 'log_IC50', k=100)

models = {
    'RF': RandomForestRegressor(random_state=42),
    'GBR': GradientBoostingRegressor(random_state=42),
    'XGB': XGBRegressor(random_state=42, verbosity=0)
}
params = {
    'RF': {'n_estimators': [100, 200], 'max_depth': [None, 10]},
    'GBR': {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1]},
    'XGB': {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 6]}
}

best_models = {}
for name, model in models.items():
    gs = GridSearchCV(model, params[name], cv=5, scoring='r2', n_jobs=-1)
    gs.fit(X_train, y_train)
    best_models[name] = gs.best_estimator_
    y_pred_log = gs.predict(X_test)
    y_true = np.expm1(y_test)
    y_pred = np.expm1(y_pred_log)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f'{name}: R²={r2:.3f}, MAE={mae:.3f}, RMSE={rmse:.3f}')