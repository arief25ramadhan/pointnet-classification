import os
import glob
import trimesh
import numpy as np
from tensorflow import data as tf_data
from keras import ops
import keras
from keras import layers
from matplotlib import pyplot as plt

# Load dataset
data = POINTCLOUD_DATA()
train_points, test_points, train_labels, test_labels, CLASS_MAP = data.parse_dataset()
train_dataset, validation_dataset, test_dataset = data.get_dataset(train_points, test_points, train_labels, test_labels)

# Load model
pointnet = POINTNET_MODEL()
model = pointnet.get_model()

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["sparse_categorical_accuracy"],
)

# Train model
checkpoint_path = "model/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                save_weights_only=True,
                                                verbose=1)

model.fit(train_dataset, epochs=20, validation_data=validation_dataset, callbacks=[cp_callback])