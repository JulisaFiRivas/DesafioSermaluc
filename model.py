from flask import Flask, request, render_template
import xgboost as xgb
import json

app = Flask(__name__)

# Cargar el modelo XGBoost
def load_model():
    with open('modelo_xgboost.json', 'r') as model_file:
        model_json = json.load(model_file)
    model = xgb.Booster()
    model.load_model('modelo_xgboost.json')
    return model

model = load_model()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Obtener datos del formulario
        features = [float(request.form[f'feature_{i}']) for i in range(1, 5)]  # Ajusta el rango según el número de características que necesites
        
        # Preparar los datos para la predicción
        dmatrix = xgb.DMatrix([features])
        
        # Realizar la predicción
        prediction = model.predict(dmatrix)[0]
        
        return render_template('result.html', prediction=prediction)
    
    return render_template('form.html')

if __name__ == '__main__':
    app.run(debug=True)