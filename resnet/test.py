import tensorflow as tf
import numpy as np
import resnet_model
import pickle

testing_file = './data/test.p'

with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

x_test, y_test = test['features'].astype(np.float32), test['labels']

with tf.Session() as sess:

	img = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
	labels = tf.placeholder(tf.int32, shape=[None, ])

	saver = tf.train.import_meta_graph('my-model.meta')
	saver.restore(sess, tf.train.latest_checkpoint('./'))
	cost = tf.get_collection('cost')[0]
	print(sess.run(cost), feed_dict = {img: x_test, labels: y_test})
