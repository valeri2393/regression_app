import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

st.title("Логистическая регрессия и визуализация данных")

uploaded_file = st.file_uploader("Загрузите файл .csv", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Пример данных:")
    st.write(df.head())

    target_column = st.selectbox("Выберите колонку с целевой переменной", df.columns)
    feature_columns = st.multiselect("Выберите колонки с признаками", [col for col in df.columns if col != target_column])

    if target_column and feature_columns:
        X = df[feature_columns]
        y = df[target_column]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LogisticRegression()
        model.fit(X_scaled, y)

        coef_dict = {feature: coef for feature, coef in zip(feature_columns, model.coef_[0])}
        st.write("Вес признаков (коэффициенты логистической регрессии):")
        st.write(coef_dict)

        st.write("Построение scatter plot")
        scatter_x = st.selectbox("Выберите X для scatter plot", feature_columns)
        scatter_y = st.selectbox("Выберите Y для scatter plot", feature_columns)

        if scatter_x and scatter_y:
            fig, ax = plt.subplots()
            scatter = ax.scatter(df[scatter_x], df[scatter_y], c=df[target_column], cmap='viridis')
            legend = ax.legend(*scatter.legend_elements(), title=target_column)
            ax.add_artist(legend)
            plt.xlabel(scatter_x)
            plt.ylabel(scatter_y)
            st.pyplot(fig)
