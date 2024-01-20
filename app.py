# librerías necesarias para le manejo de tensorflow y la comunicación del modelo y la página web
import numpy as np
import os
import tensorflow as tf
import tensorflow_addons as tfa
from keras.models import load_model
from keras import backend as K
#from werkzeug import secure_filename
#from keras.preprocessing.image import ImageDataGenerator
from flask import Flask, render_template, request, redirect, url_for, jsonify 


# nombre de nuestro servidor
app = Flask(__name__)

radam = tfa.optimizers.RectifiedAdam()
ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)

longitud, altura = 150, 150

#rutas
'''
MODELO = "/home/rodol/sis/modelo/modelo.h5"
PESOS_MODELO = "/home/rodol/sis/modelo/pesos.h5"
'''
MODELO = "D:/sis/modelo/modelo.h5"
PESOS_MODELO = "D:/sis/modelo/pesos.h5"


#funciones con los párámetros con los que estamos trabajando el modelo 
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

#Cargamos el modelo
cnn = load_model(
    MODELO,
    custom_objects={"f1_m": f1_m, "recall_m": recall_m, "precision_m": precision_m},
)

#cargamos los pesos
cnn.load_weights(PESOS_MODELO)

#aqui definimos como funciona cadá página de la página web

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        return redirect(url_for("index"))

    return render_template("index.html")


@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
            return redirect(url_for("index"))

    return render_template("contact.html")


@app.route("/portfolio", methods=["GET", "POST"])
def portfolio():
    if request.method == "POST":
        return redirect(url_for("index"))

    return render_template("portfolio.html")

@app.route("/services", methods=["GET", "POST"])
def services():
    if request.method == "POST":
        labels = ["Mildiu velloso", "hojas sanas", "moho gris", "verticilosis"]
        # Obtiene la imagen enviada por el usuario
        image = request.files["image"]
        filename = image.filename
        filepath = os.path.join("static/img/", filename)
        image.save(filepath)
        
        # Convierte la imagen en un tensor para poder realizar la predicción
        image = tf.io.read_file(filepath)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [150, 150])
        image = np.array(image, dtype=np.float32) / 255.0
        image = np.expand_dims(image, axis=0)

        # Realiza la predicción
        val = cnn.predict(image)

        # Convierte la predicción en una cadena para poder enviarla a la página HTML
        prediction = labels[np.argmax(val, axis=-1)[0]]
        prediction = str(prediction)

        # Renderiza el mismo template index.html y pasa los datos de la predicción y la imagen
        return render_template("services.html", prediction=prediction, image_url=filepath)

    os.remove(filepath)
    return render_template("services.html", prediction="", image_url="")
    

@app.route("/about", methods=["GET", "POST"])
def about():
    if request.method == "POST":
        return redirect(url_for("index"))

    return render_template("about.html")


if __name__ == "__main__":
    #app.run(host='0.0.0.0')
    app.run(debug=True)
'''    
if name == 'main':
    serve(app, host='0.0.0.0', port=80)
'''

# comando para correr el servidor
# python -u "D:\modular\servidor_SIS\app.py" python3 -m flask run
#debo abrir este archivo desde la carpeta de seridor sis en esa ruta para que funcione correctamente y cargue las rutas
