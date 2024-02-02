import os
import glob
import trimesh
import numpy as np
from tensorflow import data as tf_data
from keras import ops
import keras
from keras import layers
from matplotlib import pyplot as plt

class POINTCLOUD_DATA:

    def __init__(self, num_points=2048, num_classes=10, batch_size=32):
        self.data_dir = keras.utils.get_file("modelnet.zip",
        "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip",
        extract=True,
        )
        self.data_dir = os.path.join(os.path.dirname(self.data_dir), "ModelNet10")
        self.num_points = num_points
        self.num_classes = num_classes
        self.batch_size = batch_size

    def parse_dataset(self):
        train_points = []
        train_labels = []
        test_points = []
        test_labels = []
        class_map = {}
        folders = glob.glob(os.path.join(self.data_dir, "[!README]*"))

        for i, folder in enumerate(folders):
            print("processing class: {}".format(os.path.basename(folder)))
            # store folder name with ID so we can retrieve later
            class_map[i] = folder.split("/")[-1]
            # gather all files
            train_files = glob.glob(os.path.join(folder, "train/*"))
            test_files = glob.glob(os.path.join(folder, "test/*"))

            for f in train_files:
                train_points.append(trimesh.load(f).sample(self.num_points))
                train_labels.append(i)

            for f in test_files:
                test_points.append(trimesh.load(f).sample(self.num_points))
                test_labels.append(i)

        return (
            np.array(train_points),
            np.array(test_points),
            np.array(train_labels),
            np.array(test_labels),
            class_map,
        )

    def augment(self, points, label):
        # jitter points
        points += keras.random.uniform(points.shape, -0.005, 0.005, dtype="float64")
        # shuffle points
        points = keras.random.shuffle(points)
        return points, label


    def get_dataset(self, train_points, test_points, train_labels, test_labels, train_size=0.8):
        
        dataset = tf_data.Dataset.from_tensor_slices((train_points, train_labels))
        test_dataset = tf_data.Dataset.from_tensor_slices((test_points, test_labels))
        train_dataset_size = int(len(dataset) * train_size)

        dataset = dataset.shuffle(len(train_points)).map(self.augment)
        test_dataset = test_dataset.shuffle(len(test_points)).batch(self.batch_size)

        train_dataset = dataset.take(train_dataset_size).batch(self.batch_size)
        validation_dataset = dataset.skip(train_dataset_size).batch(self.batch_size)

        return train_dataset, validation_dataset, test_dataset

# Load dataset
data = POINTCLOUD_DATA()
train_points, test_points, train_labels, test_labels, CLASS_MAP = data.parse_dataset()
train_dataset, validation_dataset, test_dataset = data.get_dataset(train_points, test_points, train_labels, test_labels)