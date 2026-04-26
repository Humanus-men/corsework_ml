import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import load_and_prepare, classification_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
DATA_PATH = os.path.join(PROJECT_DIR, 'data', 'Данные_для_курсовои_Классическое_МО.xlsx')

df_orig, df, all_features = load_and_prepare(DATA_PATH)
median_si = df['SI'].median()
df['target'] = (df['SI'] > 8).astype(int)

# Исключаем не-признаки
exclude = ['IC50, mM', 'CC50, mM', 'SI', 'log_IC50', 'log_CC50', 'log_SI', 'target']
feat_list = [c for c in all_features if c not in exclude]

X_train, X_test, y_train, y_test, scaler = classification_split(df, 'target', feat_list)

models_clf = {
    'LR': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'RF': RandomForestClassifier(class_weight='balanced', random_state=42),
    'XGB': XGBClassifier(scale_pos_weight=(y_train==0).sum()/y_train.sum(), random_state=42)
}
params_clf = {
    'LR': {'C': [0.1, 1, 10]},
    'RF': {'n_estimators': [100, 200], 'max_depth': [None, 10]},
    'XGB': {'n_estimators': [100, 200], 'max_depth': [3, 6]}
}

for name, clf in models_clf.items():
    gs = GridSearchCV(clf, params_clf[name], cv=5, scoring='roc_auc')
    gs.fit(X_train, y_train)
    y_prob = gs.predict_proba(X_test)[:, 1]
    y_pred = gs.predict(X_test)
    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f'{name}: AUC={auc:.3f}, Acc={acc:.3f}, F1={f1:.3f}')