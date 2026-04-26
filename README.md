# Прогнозирование биологической активности химических соединений

Курсовая работа по машинному обучению.

**Задачи:** регрессия IC50, CC50, SI; классификация превышения медиан и SI > 8.

## Структура

coursework_ml/

├── data/

│   └── Данные_для_курсовои_Классическое_МО.xlsx   # исходные данные (добавьте вручную)

├── regression/

│   ├── reg_ic50.py           # регрессия IC50

│   ├── reg_cc50.py           # регрессия CC50

│   └── reg_si.py             # регрессия SI

├── classification/
│   ├── clf_ic50_median.py    # IC50 > медианы

│   ├── clf_cc50_median.py    # CC50 > медианы

│   ├── clf_si_median.py      # SI > медианы

│   └── clf_si_gt8.py         # SI > 8

├── preprocessing.py          # общие функции загрузки и подготовки данных

├── eda.py                    # разведочный анализ

├── requirements.txt          # зависимости

└── README.md

## Установка
1. Клонируйте репозиторий
2. Создайте виртуальное окружение и активируйте его
3. Установите зависимости: `pip install -r requirements.txt`
4. Поместите Excel-файл в `data/`

## Запуск
```bash
python eda/eda.py                         # EDA
python regression/reg_ic50.py            # регрессия IC50
python regression/reg_cc50.py            # регрессия CC50
python regression/reg_si.py              # регрессия SI
python classification/clf_ic50_median.py  # классификация
... (остальные аналогично)
