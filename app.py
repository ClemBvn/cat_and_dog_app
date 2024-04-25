from flask import Flask, request, render_template, redirect, url_for
import numpy as np
from keras.models import load_model
import os
from io import BytesIO
from PIL import Image
import base64
import keras

import flask_monitoringdashboard as dashboard
import mlflow
import logging

app = Flask(__name__)

# Configuration/initialisation du monitoring
dashboard.config.init_from(file='config.cfg')

# Configuration du journal de logging
logging.basicConfig(filename='error.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialisation de MLFlow
mlflow.set_tracking_uri("http://127.0.0.1:5000") # pour démarrer le serveur: $ mlflow ui
mlflow.set_experiment("dog_cat_classifier")

MODEL_PATH = os.path.join("models", "model.keras")

model = load_model(MODEL_PATH)
model.make_predict_function()

def model_predict(img, model):
    x = keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    x = np.resize(x, (1, 128, 128, 3))
    preds = model.predict(x)
    return preds

# Gestionnaire d'erreurs global (pour les erreurs non gérées)
@app.errorhandler(Exception)
def handle_error(error):
    logging.exception(f'Unhandled exception: {error}', exc_info=True)
    return 'Internal Server Error', 500

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/result', methods=['GET', 'POST'])
def upload():
    try:
        if request.method == 'POST':
            f = request.files['file']
            
            buffered_img = BytesIO(f.read())
            img = Image.open(buffered_img)

            base64_img = base64.b64encode(buffered_img.getvalue()).decode("utf-8")

            preds = model_predict(img, model)
            result = "Chien" if preds[0][0] < 0.5 else "Chat"
            
            # Sauvegarde de l'image sur le disque
            image_path = "image_uploaded.png"
            img.save(image_path)
            
            # Réception du feedback
            feedback = request.form.get('feedback')
            if feedback is not None:
                feedback = int(feedback)

            # Enregistrement de la prédiction dans MLFlow
            with mlflow.start_run() as run:
                mlflow.log_param("prediction", result)
                mlflow.log_param("prediction_probabilities", preds.tolist())
                mlflow.log_artifact(image_path, "image")
                
                # Récupération de l'ID d'exécution MLFlow
                run_id = run.info.run_id

            return render_template('result.html', result=result, image_base64_front=base64_img, run_id=run_id)
        
        return redirect('/')
    
    except Exception as e:
        logging.exception(f"An error occurred: {str(e)}")
        return "Internal Server Error", 500

# Route pour enregistrer un feedback positif dans MLflow
@app.route('/feedback/positive/<string:run_id>', methods=['GET'])
def feedback_positive(run_id):
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("feedback", 1)
    return redirect('/')

# Route pour enregistrer un feedback négatif dans MLflow
@app.route('/feedback/negative/<string:run_id>', methods=['GET'])
def feedback_negative(run_id):
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("feedback", 0)
    return redirect('/')

# Monitoring de l'app
dashboard.bind(app)

if __name__ == '__main__':
    app.run(debug=True, port=5001)