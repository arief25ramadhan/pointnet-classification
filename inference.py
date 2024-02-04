import os
import glob
import pickle
import trimesh
import numpy as np
import tensorflow as tf
from tensorflow import data as tf_data
import keras
from keras import layers
import keras_core
from keras_core import ops
from matplotlib import pyplot as plt
from dataset import POINTCLOUD_DATA
from model import POINTNET_MODEL, OrthogonalRegularizer

# Load model
print("Load Model")
pointnet = POINTNET_MODEL()
model = pointnet.get_model()
model_weights_path = 'model/pointnet.weights.h5'
model.load_weights(model_weights_path)

test_path = 'dataset/ModelNet10/toilet/test/toilet_0428.off'
points = np.array([trimesh.load(test_path).sample(2048)])

# Prediction
preds = model.predict(points)
preds = ops.argmax(preds, -1)

# Load class map
with open('model/class_map.pkl', 'rb') as handle:
    CLASS_MAP = pickle.load(handle)

label = (os.path.basename(test_path)).split('_')[0]

print("Label: ", label)
print("Prediction: ", CLASS_MAP[preds[0].numpy()])

# Plot image
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(projection="3d")
ax.scatter(points[0, :, 0], points[0, :, 1], points[0, :, 2])
ax.set_title("label: {} \n prediction: {}".format(label, CLASS_MAP[preds[0].numpy()]))
ax.set_axis_off()
plt.savefig('assets/test_inference.png')
# plt.show()