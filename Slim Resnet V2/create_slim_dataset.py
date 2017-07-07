import tensorflow as tf
from datasets import flowers

slim = tf.contrib.slim

# Selects 'validation' dataset.
dataset = flowers.get_split('validation', './flowers_data')

# Creates a TF-Slim DataProvider which reads the dataset in the background
# during both training and testing
provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
[image, label] = provider.get(['image', 'label'])