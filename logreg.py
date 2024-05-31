import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    st.title("Линейная Регрессия")

    # Загрузка файла .csv
    uploaded_file = st.file_uploader("Загрузите файл .csv", type=['csv'])
    if uploaded_file is not None:
        # Чтение данных из файла
        data = pd.read_csv(uploaded_file)

        # Вывод первых нескольких строк данных
        st.subheader('Yесколько строк данных:')
        st.write(data.head())

        # Преобразование категориальных переменных
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns
        data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

        # Выбор признаков и целевой переменной
        st.sidebar.header("Настройки регрессии")
        features = st.sidebar.multiselect("Выберите признаки для регрессии:", data.columns)
        target = st.sidebar.selectbox("Выберите целевую переменную:", data.columns)

        # Регрессия
        if st.sidebar.button("Выполнить регрессию"):
            if len(features) == 0:
                st.sidebar.error("Выберите хотя бы один признак.")
            else:
                X = data[features]
                y = data[target]
                model = LinearRegression()
                model.fit(X, y)
                coefficients = dict(zip(features, model.coef_))
                coefficients['Intercept'] = model.intercept_
                st.subheader("Результат регрессии:")
                st.json(coefficients)

        # Scatter plot
        st.sidebar.header("Настройки Scatter Plot")
        scatter_cols = st.sidebar.multiselect("Выберите два признака для построения Scatter Plot:", data.columns)
        point_size = st.sidebar.slider("Выберите размер точек:", 10, 200, 50)
        if len(scatter_cols) == 2:
            if st.sidebar.button("Построить Scatter Plot"):
                fig, ax = plt.subplots(figsize=(10, 6))
                scatter = ax.scatter(data[scatter_cols[0]], data[scatter_cols[1]], c=data[target], s=point_size, alpha=0.6, linewidth=0.5)
                ax.set_xlabel(scatter_cols[0])
                ax.set_ylabel(scatter_cols[1])
                ax.set_title("Scatter Plot")
                st.pyplot(fig)

if __name__ == "__main__":
    main()
