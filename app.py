from flask import Flask, request, render_template, flash
import xgboost as xgb
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
app.secret_key = 'mi_clave_secreta'  # Necesario para flash messages

def cargar_modelo_y_transformaciones():
    """Carga el modelo XGBoost y las transformaciones guardadas"""
    try:
        # Cargar modelo
        modelo = xgb.Booster()
        modelo.load_model('modelo_xgboost.json')
        
        # Cargar transformaciones
        with open('transformaciones.pkl', 'rb') as f:
            transformaciones = pickle.load(f)
            
        return modelo, transformaciones
    except Exception as e:
        print(f"Error al cargar modelo o transformaciones: {str(e)}")
        return None, None

# Cargar modelo y transformaciones al iniciar la aplicación
modelo, transformaciones = cargar_modelo_y_transformaciones()

def procesar_entrada(datos_formulario):
    """Procesa y transforma los datos de entrada según las transformaciones guardadas"""
    try:
        # Crear diccionario para almacenar los datos procesados
        datos_procesados = {}
        
        # Procesar cada campo
        for columna, valor in datos_formulario.items():
            if columna in transformaciones:
                # Si es una columna categórica, aplicar transformación
                if valor not in transformaciones[columna]:
                    # Si el valor no existe en las transformaciones, manejarlo
                    flash(f"Advertencia: El valor '{valor}' para '{columna}' no está en los datos de entrenamiento.")
                    datos_procesados[columna] = -1  # Valor por defecto para categorías desconocidas
                else:
                    datos_procesados[columna] = transformaciones[columna][valor]
            else:
                # Si es numérica, convertir a float
                try:
                    datos_procesados[columna] = float(valor)
                except ValueError:
                    raise ValueError(f"Error al convertir el valor '{valor}' para la columna '{columna}'")
        
        # Asegurar que todas las columnas necesarias estén presentes
        columnas_esperadas = list(transformaciones.keys())  # Añadir aquí columnas numéricas si es necesario
        for columna in columnas_esperadas:
            if columna not in datos_procesados:
                raise ValueError(f"Falta la columna requerida: {columna}")
        
        return datos_procesados
    except Exception as e:
        raise ValueError(f"Error al procesar los datos: {str(e)}")

@app.route('/', methods=['GET', 'POST'])
def index():
    if modelo is None or transformaciones is None:
        return "Error: No se pudo cargar el modelo o las transformaciones", 500
        
    if request.method == 'POST':
        try:
            # Procesar datos del formulario
            datos_formulario = request.form.to_dict()
            datos_procesados = procesar_entrada(datos_formulario)
            
            # Convertir a formato adecuado para XGBoost
            # Asegurar orden correcto de características
            columnas_ordenadas = sorted(datos_procesados.keys())
            features = [datos_procesados[col] for col in columnas_ordenadas]
            
            # Crear DMatrix y realizar predicción
            dmatrix = xgb.DMatrix([features])
            prediccion = modelo.predict(dmatrix)[0]
            
            # Renderizar el template con la predicción
            return render_template('registro.html', 
                                   prediccion=prediccion,
                                   datos_originales=datos_formulario,
                                   datos_procesados=datos_procesados)
            
        except Exception as e:
            flash(f"Error al procesar la solicitud: {str(e)}")
            return render_template('registro.html', 
                                   transformaciones=transformaciones,
                                   valores_posibles={col: list(trans.keys()) 
                                                     for col, trans in transformaciones.items()})
    
    # Para solicitudes GET, mostrar el formulario
    return render_template('registro.html', 
                           transformaciones=transformaciones,
                           valores_posibles={col: list(trans.keys()) 
                                             for col, trans in transformaciones.items()})

if __name__ == '__main__':
    app.run(debug=True)