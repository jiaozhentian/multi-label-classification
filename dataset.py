import os
import io
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE
import tqdm
import pandas as pd
import numpy as np
import PIL.Image as Image
import tensorflow as tf
from sklearn.model_selection import train_test_split

class LoadData(object):
    def __init__(self, image_path, label_path, image_size):
        self.image_path = image_path
        self.label_path = label_path
        self.image_size = image_size

    def extract_data(self, file_path):
        data = pd.read_csv(file_path)
        head = list(data.columns)
        return head, data

    def read_image(self, file_path):
        image = tf.io.read_file(file_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [self.image_size, self.image_size])
        return image

    def image_verify(self, image_path):
        try:
            image = Image.open(image_path)
            image.verify()
            return True
        except:
            return False
        
    def load_data(self):
        ds_head, ds_data = self.extract_data(self.label_path)
        file_list = ds_data[ds_head[0]].tolist()
        curpath = os.path.dirname(os.path.realpath(__file__))
        file_path_list = [os.path.join(curpath, self.image_path, f) for f in file_list]
        # verify images
        print('>'*5, "Verifying images:")
        for image_path in tqdm.tqdm(file_path_list):
            if not self.image_verify(image_path):
                print("{} is not a valid image".format(image_path))
        # Get the classes number
        num_classes = len(ds_head[1:])
        labels = ds_data[ds_head[1:]].values

        # Split the dataset into training and testing sets
        train_file_path_list, val_file_path_list, train_labels, val_labels = train_test_split(file_path_list, labels, test_size=0.2, random_state=42)
        # Create a dataset containing the filenames of the training images
        print('>'*5, 'The train file length is: ', len(train_file_path_list))
        print('>'*5, 'The test file length is: ', len(val_file_path_list))
        # Set all negative labels to 0
        train_labels[train_labels == -1] = 0
        val_labels[val_labels == -1] = 0
        # generate image dataset and label dataset respectively
        train_path_ds = tf.data.Dataset.from_tensor_slices(train_file_path_list)
        train_label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(train_labels, tf.int64))
        train_ds = train_path_ds.map(self.read_image, num_parallel_calls=AUTOTUNE)

        val_path_ds = tf.data.Dataset.from_tensor_slices(val_file_path_list)
        val_label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(val_labels, tf.int64))
        val_ds = val_path_ds.map(self.read_image, num_parallel_calls=AUTOTUNE)
        
        # print(train_label_ds.take(1).as_numpy_iterator().next())
        # zip the image and labels together
        train_dataset = tf.data.Dataset.zip((train_ds, train_label_ds))
        val_dataset = tf.data.Dataset.zip((val_ds, val_label_ds))
        
        return train_dataset, val_dataset, num_classes