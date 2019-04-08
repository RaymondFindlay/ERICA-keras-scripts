# Convert h5 keras model to tensorflow pb graph model
# Raymond Findlay 2019

import keras
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from keras.models import load_model
from keras import backend as K

model = load_model('<model>')
[node.op.name for node in model.outputs]

session = K.get_session()
min_graph = graph_util.convert_variables_to_constants(session, session.graph_def,[node.op.name for node in model.outputs])

tf.train.write_graph(min_graph, '<dir>', '<new model name>.pb', as_text=True)
