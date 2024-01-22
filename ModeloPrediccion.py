# Importar librerias
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

st.title('Análisis de Turismo por Región')

# Carga de datos
uploaded_file = st.file_uploader("Cargar archivo CSV", type=['csv'])
if uploaded_file is not None:
    try:
        turismo_df = pd.read_csv(uploaded_file)

        # Filtrar los datos para el motivo de viaje "Turismo" y "Entradas"
        turismo_df = turismo_df[(turismo_df['Motivo Viaje'] == 'Turismo') & (turismo_df['Entrada Salida'] == 'Entradas')]

        # Crear y entrenar un modelo para cada región
        regiones = turismo_df['Region Nacionalidad'].unique()

        # Almacenar predicciones y métricas para cada región
        predicciones_regionales = {}
        metricas_regionales = {}

        # Menú desplegable para seleccionar el modelo
        modelo_seleccionado = st.selectbox('Selecciona el modelo de análisis', ('Regresión Lineal', 'Random Forest', 'SVM', 'XGBoost'))

        # Menú desplegable para seleccionar la región
        region_seleccionada = st.selectbox('Selecciona una región para ver la predicción y métricas:', ['Todas las regiones'] + list(regiones))

        if region_seleccionada == 'Todas las regiones':
            # Sumar la cantidad total de personas de todas las regiones por año
            total_personas_por_año = turismo_df.groupby(['Año'])['Total Personas'].sum().reset_index()

            if modelo_seleccionado == 'Regresión Lineal':
                # Crear y entrenar el modelo de regresión lineal para todas las regiones
                model = LinearRegression()
                model.fit(total_personas_por_año[['Año']], total_personas_por_año['Total Personas'])

                # Realizar predicciones para el año 2024
                prediction_2024 = model.predict([[2024]])

            elif modelo_seleccionado == 'Random Forest':
                # Crear y entrenar el modelo de Random Forest para todas las regiones
                model_rf = RandomForestRegressor()
                model_rf.fit(total_personas_por_año[['Año']], total_personas_por_año['Total Personas'])

                # Realizar predicciones para el año 2024
                prediction_2024 = model_rf.predict([[2024]])

            elif modelo_seleccionado == 'SVM':
                # Crear y entrenar el modelo de SVM para todas las regiones
                model_svm = SVR()
                model_svm.fit(total_personas_por_año[['Año']], total_personas_por_año['Total Personas'])

                # Realizar predicciones para el año 2024
                prediction_2024 = model_svm.predict([[2024]])

            elif modelo_seleccionado == 'XGBoost':
                # Crear y entrenar el modelo XGBoost para todas las regiones
                model_xgb = xgb.XGBRegressor()
                model_xgb.fit(total_personas_por_año[['Año']], total_personas_por_año['Total Personas'])

                # Realizar predicciones para el año 2024
                prediction_2024 = model_xgb.predict([[2024]])

            # Gráfico de tendencias históricas y predicciones para todas las regiones
            plt.figure(figsize=(10, 6))

            # Tendencias históricas
            plt.plot(total_personas_por_año['Año'], total_personas_por_año['Total Personas'], marker='o', label=f'Todas las Regiones - Historico')

            # Predicciones agregadas
            plt.plot(2024, prediction_2024[0], marker='o', linestyle='--', color='black', label=f'Predicción 2024 - Todas las Regiones')

            # Etiquetas y título
            plt.xlabel('Año')
            plt.ylabel('Total de Personas')
            plt.title(f'Tendencias Históricas y Predicción para Todas las Regiones (Turismo - Entradas)')
            plt.legend()

            # Métricas de rendimiento
            if modelo_seleccionado == 'Regresión Lineal':
                mse = mean_squared_error(total_personas_por_año['Total Personas'], model.predict(total_personas_por_año[['Año']]))
                r2 = r2_score(total_personas_por_año['Total Personas'], model.predict(total_personas_por_año[['Año']]))
            elif modelo_seleccionado == 'Random Forest':
                mse = mean_squared_error(total_personas_por_año['Total Personas'], model_rf.predict(total_personas_por_año[['Año']]))
                r2 = r2_score(total_personas_por_año['Total Personas'], model_rf.predict(total_personas_por_año[['Año']]))
            elif modelo_seleccionado == 'SVM':
                mse = mean_squared_error(total_personas_por_año['Total Personas'], model_svm.predict(total_personas_por_año[['Año']]))
                r2 = r2_score(total_personas_por_año['Total Personas'], model_svm.predict(total_personas_por_año[['Año']]))
            elif modelo_seleccionado == 'XGBoost':
                mse = mean_squared_error(total_personas_por_año['Total Personas'], model_xgb.predict(total_personas_por_año[['Año']]))
                r2 = r2_score(total_personas_por_año['Total Personas'], model_xgb.predict(total_personas_por_año[['Año']]))

            # Mostrar la predicción y las métricas para "Todas las regiones"
            st.markdown(f"**Predicción para el año 2024 en Todas las Regiones ({modelo_seleccionado}):**")
            st.markdown(f"**{prediction_2024[0]:,.0f}** personas")
            st.markdown(f"**Métricas de Efectividad para Todas las Regiones ({modelo_seleccionado}):**")
            st.markdown(f"MSE: {mse:,.2f}")
            st.markdown(f"RMSE: {np.sqrt(mse):,.2f}")
            st.markdown(f"R²: {r2:,.2f}")

            # Mostrar el gráfico
            st.pyplot(plt)

        else:
            # Filtrar datos solo para la región seleccionada
            region_df = turismo_df[turismo_df['Region Nacionalidad'] == region_seleccionada]
            total_personas_por_año = region_df.groupby('Año')['Total Personas'].sum().reset_index()

            if modelo_seleccionado == 'Regresión Lineal':
                # Crear y entrenar el modelo de regresión lineal para la región seleccionada
                model = LinearRegression()
                model.fit(total_personas_por_año[['Año']], total_personas_por_año['Total Personas'])

                # Realizar predicciones para el año 2024
                prediction_2024 = model.predict([[2024]])

            elif modelo_seleccionado == 'Random Forest':
                # Crear y entrenar el modelo de Random Forest para la región seleccionada
                model_rf = RandomForestRegressor()
                model_rf.fit(total_personas_por_año[['Año']], total_personas_por_año['Total Personas'])

                # Realizar predicciones para el año 2024
                prediction_2024 = model_rf.predict([[2024]])

            elif modelo_seleccionado == 'SVM':
                # Crear y entrenar el modelo de SVM para la región seleccionada
                model_svm = SVR()
                model_svm.fit(total_personas_por_año[['Año']], total_personas_por_año['Total Personas'])

                # Realizar predicciones para el año 2024
                prediction_2024 = model_svm.predict([[2024]])

            elif modelo_seleccionado == 'XGBoost':
                # Crear y entrenar el modelo XGBoost para la región seleccionada
                model_xgb = xgb.XGBRegressor()
                model_xgb.fit(total_personas_por_año[['Año']], total_personas_por_año['Total Personas'])

                # Realizar predicciones para el año 2024
                prediction_2024 = model_xgb.predict([[2024]])

            # Gráfico de tendencias históricas y predicciones solo para la región seleccionada
            plt.figure(figsize=(10, 6))

            # Tendencias históricas
            plt.plot(total_personas_por_año['Año'], total_personas_por_año['Total Personas'], marker='o', label=f'{region_seleccionada} - Historico')

            # Predicciones
            plt.plot(2024, prediction_2024[0], marker='o', linestyle='--', color='red', label=f'{region_seleccionada} - Predicción 2024')

            # Etiquetas y título
            plt.xlabel('Año')
            plt.ylabel('Total de Personas')
            plt.title(f'Tendencias Históricas y Predicción para {region_seleccionada} (Turismo - Entradas)')
            plt.legend()

            # Métricas de rendimiento
            if modelo_seleccionado == 'Regresión Lineal':
                mse = mean_squared_error(total_personas_por_año['Total Personas'], model.predict(total_personas_por_año[['Año']]))
                r2 = r2_score(total_personas_por_año['Total Personas'], model.predict(total_personas_por_año[['Año']]))
            elif modelo_seleccionado == 'Random Forest':
                mse = mean_squared_error(total_personas_por_año['Total Personas'], model_rf.predict(total_personas_por_año[['Año']]))
                r2 = r2_score(total_personas_por_año['Total Personas'], model_rf.predict(total_personas_por_año[['Año']]))
            elif modelo_seleccionado == 'SVM':
                mse = mean_squared_error(total_personas_por_año['Total Personas'], model_svm.predict(total_personas_por_año[['Año']]))
                r2 = r2_score(total_personas_por_año['Total Personas'], model_svm.predict(total_personas_por_año[['Año']]))
            elif modelo_seleccionado == 'XGBoost':
                mse = mean_squared_error(total_personas_por_año['Total Personas'], model_xgb.predict(total_personas_por_año[['Año']]))
                r2 = r2_score(total_personas_por_año['Total Personas'], model_xgb.predict(total_personas_por_año[['Año']]))

            # Mostrar la predicción y las métricas para la región seleccionada
            st.markdown(f"**Predicción para el año 2024 en {region_seleccionada} ({modelo_seleccionado}):**")
            st.markdown(f"**{prediction_2024[0]:,.0f}** personas")
            st.markdown(f"**Métricas de Efectividad para {region_seleccionada} ({modelo_seleccionado}):**")
            st.markdown(f"MSE: {mse:,.2f}")
            st.markdown(f"RMSE: {np.sqrt(mse):,.2f}")
            st.markdown(f"R²: {r2:,.2f}")

            # Mostrar el gráfico
            st.pyplot(plt)

    except Exception as e:
        st.error(f"Se produjo un error al leer el archivo: {e}")








