import os
import pickle
import glob
import trimesh
import numpy as np
from tensorflow import data as tf_data
import keras_core
from keras_core import ops
import tensorflow as tf
import keras
from keras import layers
from matplotlib import pyplot as plt
from dataset import POINTCLOUD_DATA
from model import POINTNET_MODEL, OrthogonalRegularizer

# Load dataset
print("Load Data")
data = POINTCLOUD_DATA()
train_points, test_points, train_labels, test_labels, CLASS_MAP = data.parse_dataset()
train_dataset, validation_dataset, test_dataset = data.get_dataset(train_points, test_points, train_labels, test_labels)

with open('model/class_map.pkl', 'wb') as handle:
    pickle.dump(CLASS_MAP, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Load model
print("Load Model")
pointnet = POINTNET_MODEL()
model = pointnet.get_model()

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["sparse_categorical_accuracy"],
)

# Train model
print("Train Model")
checkpoint_path = "model/pointnet.weights.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                save_weights_only=True,
                                                monitor='val_sparse_categorical_accuracy',
                                                mode='max',
                                                save_best_only=True,
                                                verbose=1)

model.fit(train_dataset, epochs=20, validation_data=validation_dataset, callbacks=[cp_callback])