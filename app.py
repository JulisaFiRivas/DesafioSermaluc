from flask import Flask, request, render_template, jsonify
import pickle
import xgboost as xgb
import numpy as np
import pandas as pd
from LabelEncoderPersistence import LabelEncoderPersistence
app = Flask(__name__)


model = xgb.XGBClassifier()
model.load_model('modelo_xgboost.json')

encoder = LabelEncoderPersistence()
encoder.load_transformations('transformaciones')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Recibir los datos del formulario
        sexo = request.form.get('sexo')
        edad = request.form.get('edad')
        tipo_registro = request.form.get('tipo-registro')
        servicio_ingreso = request.form.get('servicio-ingreso')
        diagnostico = request.form.get('diagnostico')

        # Crear un diccionario con los datos
        input_data = pd.DataFrame({
            'SEXO': [sexo],
            'TIPO_INGRESO': [tipo_registro],
            'SERVICIOINGRESO': [servicio_ingreso],
            'DIAGNOSTICO1': [diagnostico]
        })

        input_data = encoder.transform_columns(input_data, input_data.columns)
        input_data['EDAD'] = int(edad)


        # Realizar la predicción
        prediction = model.predict(input_data)

        mortalidad = {
            0:"Sin gravedad",
            1:"Menor",
            2:"Moderada",
            3:"Mayor"
        }

        # Mostrar el resultado en la página
        return render_template('resultado.html', resultado=prediction[0], nivel_mortalidad=mortalidad[prediction[0]])

    return render_template('registro.html')

if __name__ == '__main__':
    app.run(debug=True)
