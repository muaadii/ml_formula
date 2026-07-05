# Фильтрация спама
# Бинарная классификация
# Векторизация

# столбцы = слова ( в тексте)
# строки = образцы текста
# ячейка = кол-во данных слов в данном тексте

# # очистка: строчные, удаление знаков препинания, стоп - слова,

# import numpy as np
# import pandas as pd

# data = pd.read_csv("digital_python-25-26/data/spam.csv")
# print(data.head())

# data["Spam"] = data["Category"].apply(lambda x: 1 if x == "spam" else 0)

# print(data.columns)

# print(data.info())

# from sklearn.feature_extraction.text import CountVectorizer

# vectorizer = CountVectorizer()
# X = vectorizer.fit_transform(data["Message"])
# w = vectorizer.get_feature_name_out()

# # print(w)
# # print(w[1000])

# # print(X)

# # print(X[:,1000])

# from sklearn.model_selection import train_test_split

# X_tr, X_tst, y_tr, y_tst = train_test_split(data["Message"], data["Spam"], test_size=0.25)

# from sklearn.naive_bayes import MultinominalNB

# from sklearn.pipeline import Pipeline

# md = Pipeline([("vectorizer", CountVectorizer()), ("nb", MultinominalNB())])

# md.fit(X_tr, y_tr)

# texts = ["Hi! How are you?",  # 0
#          "Win the lottery",   # 0
#          "Free subscription",   # 1
#          "Black Friday big discount shop offer",   # 1
#          "Nice to meet you"   # 0
#          ]

# print(md.predict(texts))


#  Фишинг

# import numpy as np
# import pandas as pd

# data = pd.read_csv("digital_python-25-26/data/phishing.csv")
# print(data.head())

# print(data.columns)

# X = data.drop(columns=["class"])
# print(X.columns)

# y = pd.DataFrame(data["class"])
# print(y.columns)

# from sklearn.model_selection import train_test_split

# X_tr, X_tst, y_tr, y_tst = train_test_split(X, y, test_size=0.25)

# from sklearn.tree import DecisionTreeClassifier

# dt = DecisionTreeClassifier()

# model = dt.fit(X_tr, y_tr)

# predict = model.predict(X_tst)

# from sklearn.metrics import accuracy_score

# print(accuracy_score(predict, y_tst))

# Классификация: бинарные(двоичные), мультиклассовые, многометочные
# - точность (precision) - стоимость ложных срабатываний высока
# - полнота (recall) - стоимость ложноотрицательных срабатываний высока
# - специфичночть (specificity) = полнота для истинноположительных, насколько точно определяются отрицательные образцы
# - чувствительность (sensitivity) = полнота
# - F1-мера

#  Метрики: - процент ошибок, процент правильных ответов (accuracy)

# Типы ошибочных ответов: ложноположительные (ложная тревога), ложноотрицательные (ложный пропуск)
# Типы правильных ответов: истинноположительные, истинноотрицательные


#  Аномалии

import numpy as np
import pandas as pd

data = pd.read_csv("digital_python-25-26/data/creditcard.csv")
print(data.head())

legit = data[data["Class"] == 0]
fraud = data[data["Class"] == 1]

X = data.drop(["Time", "Class"], axis=1)
y = data["Class"]

from sklearn.model_selection import train_test_split

X_tr, X_tst, y_tr, y_tst = train_test_split(X, y, test_size=0.25)

from sklearn.linear_model import LogisticRegression

model1 = LogisticRegression()
model1.fit(X_tr, y_tr)

import matplotlib as plt
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(
    model1, X_tst, y_tst, display_labels=["Легитимная", "Мошенническая"])

plt.show()  # 14

from sklearn.metrics import precision_score, recall_score

# Точность
y_pred = model1.predict(X_tst)
print(precision_score(y_tst, y_pred))

# Полнота
print(recall_score(y_tst, y_pred))

# Специфичность
print(recall_score(y_tst, y_pred, pos_label=0))

# from sklearn.metrics import RandomForestClassifier

# model2 = RandomForestClassifier(n_estimators=10)
# model2.fit(X_tr, y_tr)

# ConfusionMatrixDisplay.from_estimator(
#     model2, X_tst, y_tst, display_labels=["Легитимная", "Мошенническая"]
# )


# plt.show()  # ?


# from sklearn.metrics import GradientBoostingClassifier

# model3 = GradientBoostingClassifier()
# model3.fit(X_tr, y_tr)

# ConfusionMatrixDisplay.from_estimator(
#     model3, X_tst, y_tst, display_labels=["Легитимная", "Мошенническая"]
# )


# plt.show()