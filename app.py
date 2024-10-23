from flask import Flask, request, render_template, jsonify
import pickle
import xgboost as xgb
import numpy as np

app = Flask(__name__)


model = xgb.XGBClassifier()
model.load_model('modelo_xgboost.json')

with open('transformaciones.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

"""# Función para preprocesar los datos
def preprocess_input(data):
    # Transformar datos categóricos con los LabelEncoders
    for column, encoder in label_encoders.items():
        if column in data:
            data[column] = encoder.transform([data[column]])[0]
    return np.array(list(data.values())).reshape(1, -1)
"""
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
        input_data = {
            'SEXO': sexo,
            'TIPO_INGRESO': tipo_registro,
            'SERVICIOINGRESO': servicio_ingreso,
            'DIAGNOSTICO1': diagnostico
        }
        print(input_data)
       

       
        # Preprocesar los datos
        #preprocessed_data = preprocess_input(input_data)
        X_test =[[sexo,edad,tipo_registro,servicio_ingreso,diagnostico]]

        # Realizar la predicción
        prediction = model.predict(X_test)

        # Mostrar el resultado en la página
        return render_template('resultado.html', resultado=prediction[0])

    return render_template('registro.html')

if __name__ == '__main__':
    app.run(debug=True)
