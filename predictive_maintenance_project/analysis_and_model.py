import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

def analysis_and_model_page():
    st.title("Анализ данных и модель")

    # Загрузка данных
    uploaded_file = st.file_uploader("Загрузите датасет (CSV)", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # Предобработка данных
        data = data.drop(columns=['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
        data['Type'] = LabelEncoder().fit_transform(data['Type'])

        # Масштабирование числовых признаков
        scaler = StandardScaler()
        numerical_features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
        data[numerical_features] = scaler.fit_transform(data[numerical_features])

        # Разделение данных
        X = data.drop(columns=['Machine failure'])
        y = data['Machine failure']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Обучение модели
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Оценка модели
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)

        # Визуализация результатов
        st.header("Результаты обучения модели")
        st.write(f"Accuracy: {accuracy:.2f}")

        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

        st.subheader("Classification Report")
        st.text(class_report)

        # Интерфейс для предсказания
        st.header("Предсказание по новым данным")
        with st.form("prediction_form"):
            st.write("Введите значения признаков для предсказания:")
            product_type = st.selectbox("Тип продукта", ["L", "M", "H"])
            air_temp = st.number_input("Температура воздуха [K]")
            process_temp = st.number_input("Рабочая температура [K]")
            rotational_speed = st.number_input("Скорость вращения [rpm]")
            torque = st.number_input("Крутящий момент [Nm]")
            tool_wear = st.number_input("Износ инструмента [min]")

            submit_button = st.form_submit_button("Предсказать")

        if submit_button:
            input_data = pd.DataFrame({
                'Type': [LabelEncoder().fit_transform([product_type])[0]],
                'Air temperature [K]': [air_temp],
                'Process temperature [K]': [process_temp],
                'Rotational speed [rpm]': [rotational_speed],
                'Torque [Nm]': [torque],
                'Tool wear [min]': [tool_wear]
            })
            input_data[numerical_features] = scaler.transform(input_data[numerical_features])
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0][1]
            st.write(f"Предсказание: {'Отказ' if prediction == 1 else 'Нет отказа'}")
            st.write(f"Вероятность отказа: {prediction_proba:.2f}")