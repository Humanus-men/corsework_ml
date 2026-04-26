import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

def load_and_prepare(excel_path):
    """Загрузка, логарифмирование, удаление выбросов и неинформативных признаков."""
    df_original = pd.read_excel(excel_path)
    if 'Unnamed: 0' in df_original.columns:
        df_original.drop(columns=['Unnamed: 0'], inplace=True)

    df = df_original.copy()

    # Логарифмируем цели
    target_cols = ['IC50, mM', 'CC50, mM', 'SI']
    for col in ['IC50, mM', 'CC50, mM']:
        df[col] = df[col].clip(lower=1e-6)
    df['log_IC50'] = np.log1p(df['IC50, mM'])
    df['log_CC50'] = np.log1p(df['CC50, mM'])
    df['log_SI'] = np.log1p(df['SI'])

    # Удаляем выбросы по log_IC50 (IQR)
    Q1, Q3 = df['log_IC50'].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    df = df[(df['log_IC50'] >= lower) & (df['log_IC50'] <= upper)]

    # Список всех числовых признаков
    all_features = [c for c in df.columns if c not in target_cols + ['log_IC50','log_CC50','log_SI']]
    # Удаляем константные
    const_cols = [c for c in all_features if df[c].std() < 0.01]
    df.drop(columns=const_cols, inplace=True, errors='ignore')
    all_features = [c for c in all_features if c not in const_cols]

    # Удаляем строки с пропусками
    df_clean = df.dropna(subset=all_features + ['log_IC50','log_CC50','log_SI'])
    return df_original, df_clean, all_features


def regression_split(df, target, k=100, test_size=0.2, random_state=42):
    """Отбирает k лучших признаков для регрессии и возвращает обучающую/тестовую выборки."""
    exclude = ['IC50, mM', 'CC50, mM', 'SI', 'log_IC50', 'log_CC50', 'log_SI']
    feature_cols = [c for c in df.columns if c not in exclude + [target]]
    X = df[feature_cols].values
    y = df[target].values

    selector = SelectKBest(f_regression, k=min(k, len(feature_cols)))
    X_new = selector.fit_transform(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_new, y, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, selector, scaler


def classification_split(df, target_col, features_list, test_size=0.2, random_state=42):
    """Масштабирует признаки и делит выборку для классификации."""
    X = df[features_list].values
    y = df[target_col].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler