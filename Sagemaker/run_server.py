import os
import sys
import json
import numpy as np
from PIL import Image
import flask
from flask import Response
import tensorflow as tf
import io
import base64
import time

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None
session = None
def load_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global model
    global session
    # config = tf.ConfigProto(
    #     device_count={'GPU': 1},
    #     intra_op_parallelism_threads=1,
    #     allow_soft_placement=True
    # )
    # config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.6
    # session = tf.Session(config=config)
    session = tf.Session()
    tf.keras.backend.set_session(session)
    # model = tf.keras.applications.NASNetLarge(input_shape=(331, 331, 3), weights='imagenet')
    model = tf.keras.models.load_model('/models/nasnet.h5')
    model._make_predict_function()

def prepare_image(image, target):
    # resize the input image and preprocess it
    image = image.resize(target)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.nasnet.preprocess_input(image)

    # return the processed image
    return image


@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    start = time.time()

    data = {}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        content = flask.request.json
        if 'data' in content:
        # if flask.request.files.get("image"):
            # read the image in PIL format
            try:
                with session.as_default():
                    with session.graph.as_default():
                        # image = flask.request.files["image"].read()
                        image = content['data']
                        image = base64.b64decode(image)
                        image = Image.open(io.BytesIO(image))

                        # preprocess the image and prepare it for classification
                        # image = prepare_image(image, target=(331, 331))
                        image = prepare_image(image, target=(224, 224))

                        # classify the input image and then initialize the list
                        # of predictions to return to the client
                        preds = model.predict(image)
                        results = tf.keras.applications.nasnet.decode_predictions(preds, top=1)
                        data["predictions"] = str(results)
            except Exception as ex:
                error_response = {
                    'error_message': "Unexpected error",
                    'stack_trace': str(ex)
                }
                return flask.make_response(flask.jsonify(error_response), 403)

            # indicate that the request was a success
            data["latency"] = int((time.time() - start) * 1000) 

    # return the data dictionary as a JSON response
    return flask.make_response(flask.jsonify(data), 200)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_model()
    app.run()



