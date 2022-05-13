# load Flask 
import json
import os
import flask
import trimesh
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

app = flask.Flask(__name__)

class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))

    def get_config(self):
        config = {'num_features': self.num_features,
                  'l2reg': self.l2reg}
        return config


# define a predict function as an endpoint 
@app.route("/predict", methods=["GET","POST"])
def predict():
    model = tf.keras.models.load_model(os.path.join('.', 'my_model.h5'), custom_objects={"OrthogonalRegularizer" : OrthogonalRegularizer})
    CLASS_MAP = {0: 'bathtub', 1: 'bed', 2: 'chair', 3: 'night_stand', 4: 'sofa', 5: 'table', 6: 'toilet'}
    params = flask.request.json
    
    points = params["input"]

    points = np.array(points)
    # run test data through model
    preds = model.predict(points)
    preds = tf.math.argmax(preds, -1)
    print("aici")
    data = {"success": True}
    data["response"] = (CLASS_MAP[preds[0].numpy()])
    return flask.jsonify(data)
app.run()
