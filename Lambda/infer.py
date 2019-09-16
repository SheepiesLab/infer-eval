import time

start = time.time()
import tensorflow as tf
import json
import numpy as np
import io
from PIL import Image
import base64
import os
print('import time: ' + str(int((time.time() - start) * 1000)))

start = time.time()

with open('/opt/model_architecture.json', 'r') as f:
    model = tf.keras.models.model_from_json(f.read())

# Load weights into the new model
model.load_weights('/opt/model_weights.h5')

print('model time: ' + str(int((time.time() - start) * 1000)))

def lambda_handler(event, context):
    start = time.time()
    
    body = json.loads(event['body'])
    image = body['data']
    
    image = base64.b64decode(image)
    image = Image.open(io.BytesIO(image))

    image = image.resize((224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.nasnet.preprocess_input(image)

    preds = model.predict(image)
    results = tf.keras.applications.nasnet.decode_predictions(preds, top=1)
    res = {}
    res["predictions"] = str(results)
    res["latency"] = int((time.time() - start) * 1000) 

    response = {
        "statusCode": 200,
        "body": json.dumps(res)
    }

    return response