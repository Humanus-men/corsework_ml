import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import load_and_prepare
import os
import sys

# Загружаем данные
df_orig, df, features = load_and_prepare(r"data/Данные_для_курсовои_Классическое_МО.xlsx")

# Распределения логарифмированных целей
fig, ax = plt.subplots(1,3,figsize=(15,4))
for i, col in enumerate(['log_IC50','log_CC50','log_SI']):
    sns.histplot(df[col], kde=True, ax=ax[i])
    ax[i].set_title(col)
plt.tight_layout()
plt.savefig('target_distributions.png')
plt.show()

# Корреляции с log_IC50
corr = df[features + ['log_IC50']].corr()['log_IC50'].abs().sort_values(ascending=False)
print("Топ-10 корреляций с log_IC50:\n", corr.head(11))
# Сохраните данные для отчёта

# Матрица корреляций (для отчёта можно сохранить)
sample_feats = corr.index[1:21]  # топ-20 признаков
plt.figure(figsize=(12, 10))
sns.heatmap(df[sample_feats].corr(), annot=False, cmap='coolwarm')
plt.title('Корреляция 20 наиболее связанных с IC50 признаков')
plt.tight_layout()
plt.savefig('corr_heatmap.png')
plt.show()