import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

df = pd.DataFrame(pd.read_excel('D:\data7.xlsx'))
read_file = pd.read_excel('D:\data7.xlsx')

read_file.to_csv('data7.csv',
                index = None,
                header = True)
train_df = pd.read_csv('data7.csv')
train_df.head(10)

predictors = ['сложность освоения', 'уровень подготовки']
outcome = 'Класс'

new_record = train_df.loc[0:0, predictors]
X = train_df.loc[1:, predictors]
y = train_df.loc[1:, outcome]

knn = KNeighborsClassifier(n_neighbors=40)
knn.fit(X, y)
knn.predict(new_record)
knn.predict_proba(new_record)

nbrs = knn.kneighbors(new_record)
nbr_df = pd.DataFrame({'сложнность освоения': X.iloc[nbrs[1][0], 0],
                      'уровень подготовки': X.iloc[nbrs[1][0], 1],
                      'Класс': y.iloc[nbrs[1][0]]})

import streamlit as st

st.sidebar.header('Ввод параметров')

def user_input_features():
    level = st.sidebar.slider('уровень подготовки', 1, 3, 2)
    complexity = st.sidebar.slider('сложность освоения', 0.01, 1.00, 0.43)
    data = {'level': level,
           'complexity': complexity}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.write(''''''
        # Приложение для поддержки принятия решения о выборе формата тренировки **Training format**
        '''''')

st.write(df)

st.subheader('Формат тренировки и соответствующие его характеристики')

st.write(nbr_df)

st.subheader('Prediction')

st.write(nbr_df[knn.predict])
