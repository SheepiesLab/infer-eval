import tensorflow as tf
import json
import numpy as np
import io


# model = tf.keras.applications.NASNetLarge(input_shape=(331, 331, 3), weights='imagenet')
# model = tf.keras.applications.inception_v3.InceptionV3(input_shape=(224, 224, 3), weights='imagenet')

model = tf.keras.applications.NASNetMobile(input_shape=(224, 224, 3), weights='imagenet')

model.save('nasnet.h5')

# model = tf.keras.models.load_model('nasnet.h5')
