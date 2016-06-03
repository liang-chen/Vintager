
"""
Convolutional Neural Network
Including training and testing functions using TensorFlow
"""


import tensorflow as tf
from tensorflow.contrib import skflow
import numpy as np
import os
import cv2
from globv import uni_image_pixels, symbol_label_list, uni_size, cnn_model_path
from feature import pixel_vec


def weight_variable(shape):
    """
    Create tensorflow convolutional layer weights
    :param shape: shape of tensor
    :return: variable W
    :rtype: tf.Variable
    """
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """
    Create tensorflow convolutional layer bias
    :param shape: shape of tensor
    :return: variable b
    :rtype: tf.Variable
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    """
    Performing 2d convolution
    :param x: input tensor
    :param W: weight
    :return: output tensor
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    """
    Performing 2*2 max pooling
    :param x: input tensor
    :return: outpur tensor
    """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def max_pool_4x4(x):
    """
    Performing 4*4 convolution
    :param x: input tensor
    :return: outpur tensor
    """
    return tf.nn.max_pool(x, ksize=[1, 4, 4, 1],
                        strides=[1, 4, 4, 1], padding='SAME')


def get_data_batch(i, batch_size):
    cnt = 0
    n_labels = len(symbol_label_list)
    imgs = np.empty((0, uni_image_pixels), dtype = 'float32')
    labels = np.zeros((batch_size, n_labels), dtype='float32')
    circ = 0

    while cnt < batch_size:
        cur_label_index = circ % n_labels
        circ += 1

        label_name = symbol_label_list[cur_label_index]
        dir = "../data/" + label_name + "/"
        if not os.path.exists(dir):
            continue

        file_cnt = len([f for f in os.listdir(dir) if f.endswith('.jpg') and os.path.isfile(os.path.join(dir, f))])
        if file_cnt == 0:
            continue

        labels[cnt, cur_label_index] = 1
        picked_index = i % file_cnt
        img = cv2.imread(dir + str(picked_index + 1) + ".jpg", 0)
        imgs = np.append(imgs, np.array([pixel_vec(img)]), axis = 0)
        cnt += 1


    batch = (imgs, labels)
    #print np.shape(imgs)
    return batch


def get_all_data():
    n_labels = len(symbol_label_list)
    imgs = np.empty((0, uni_image_pixels), dtype='float32')
    labels = np.empty((0, n_labels), dtype='float32')

    for i in xrange(n_labels):
        label_name = symbol_label_list[i]
        dir = "../data/" + label_name + "/"
        if not os.path.exists(dir):
            continue

        file_cnt = len([f for f in os.listdir(dir) if f.endswith('.jpg') and os.path.isfile(os.path.join(dir, f))])
        if file_cnt == 0:
            continue

        for j in xrange(file_cnt):
            label_vec = np.zeros(n_labels, dtype='float32')
            label_vec[i] = 1  ##one-hot encoding
            labels = np.append(labels, np.array([label_vec]), axis = 0)
            img = cv2.imread(dir + str(j + 1) + ".jpg", 0)
            imgs = np.append(imgs, np.array([pixel_vec(img)]), axis=0)

    data = (imgs, labels)

    return data


def get_all_data_var():
    n_labels = len(symbol_label_list)
    imgs = np.empty((0, uni_image_pixels), dtype='float32')
    labels = np.array([], dtype='float32')

    for i in xrange(n_labels):
        label_name = symbol_label_list[i]
        dir = "../data/" + label_name + "/"
        if not os.path.exists(dir):
            continue

        file_cnt = len([f for f in os.listdir(dir) if f.endswith('.jpg') and os.path.isfile(os.path.join(dir, f))])
        if file_cnt == 0:
            continue

        for j in xrange(file_cnt):
            labels = np.append(labels, i)
            img = cv2.imread(dir + str(j + 1) + ".jpg", 0)
            imgs = np.append(imgs, np.array([pixel_vec(img)]), axis=0)

    data = (imgs, labels)

    return data


def conv_model(X,y):
    X = tf.reshape(X, [-1, uni_size[0], uni_size[1], 1])
    y = tf.reshape(y, [-1, len(symbol_label_list)])
    #first conv layer
    with tf.variable_scope('conv_layer1'):
        h_conv1 = skflow.ops.conv2d(X, n_filters = 32, filter_shape = [5,5],
                                    bias = True, activation = tf.nn.relu)
        h_pool1 = max_pool_4x4(h_conv1)

    # second conv layer will compute 64 features for each 5x5 patch
    with tf.variable_scope('conv_layer2'):
        h_conv2 = skflow.ops.conv2d(h_pool1, n_filters=64, filter_shape=[5, 5],
                                    bias=True, activation=tf.nn.relu)
        h_pool2 = max_pool_4x4(h_conv2)
        # reshape tensor into a batch of vectors
        h_pool2_flat = tf.reshape(h_pool2, [-1, 4 * 4 * 64])

    # densely connected layer with 1024 neurons
    h_fc1 = skflow.ops.dnn(h_pool2_flat, [1024], activation=tf.nn.relu, dropout=0.5)
    return skflow.models.logistic_regression(h_fc1, y)


def train_cnn_var():
    n_labels = len(symbol_label_list)
    # Training and predicting
    classifier = skflow.TensorFlowEstimator(
        model_fn=conv_model, n_classes=n_labels, batch_size=100, steps=20000,
        learning_rate=0.01)
    data = get_all_data_var()
    #print np.shape(data[0]), np.shape(data[1])
    classifier.fit(data[0], data[1])
    classifier.save(cnn_model_path)


def train_cnn():
    batch_size = 50
    n_labels = len(symbol_label_list)
    x = tf.placeholder(tf.float32, shape=(None, uni_image_pixels))
    y_ = tf.placeholder(tf.float32, shape=(None, n_labels))

    sess = tf.InteractiveSession()
    x_image = tf.reshape(x, [-1, uni_size[0], uni_size[1], 1])

    ##first convolutional layer
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_4x4(h_conv1)

    ##second convolutional layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_4x4(h_conv2)

    ##third convolutional layer
    # W_conv3 = weight_variable([5, 5, 64, 128])
    # b_conv3 = bias_variable([128])
    # h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    # h_pool3 = max_pool_4x4(h_conv3)

    ##fully connected layer
    W_fc1 = weight_variable([4 * 4 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 4 * 4 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    ##fully connected layer
    # W_fc2 = weight_variable([2048, 1024])
    # b_fc2 = bias_variable([1024])
    # h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

    ##Readout layer
    W_fc3 = weight_variable([1024, n_labels])
    b_fc3 = bias_variable([n_labels])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1, W_fc3) + b_fc3)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    find_y_conv = tf.argmax(y_conv,1)
    sess.run(tf.initialize_all_variables())

    iter = 40
    for i in range(iter):
        #print i
        batch = get_data_batch(i,batch_size)

        if i % 20 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1]})
            #print batch[1]
            #cv2.imshow("test", batch[0][0].reshape(64,64))
            #cv2.waitKey(0)
            print find_y_conv.eval(feed_dict={
                x: batch[0], y_: batch[1]})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})

    saver = tf.train.Saver()
    save_path = saver.save(sess, "../models/cnn_model", global_step=iter)
    print("Model saved in file: %s" % save_path)

    test_data = get_all_data()
    print("test accuracy %g" % accuracy.eval(feed_dict={
        x: test_data[0], y_: test_data[1]}))
    sess.close()


def detect_cnn(img, classifier):
    pred_index = classifier.predict(np.array([pixel_vec(img)], dtype = "float32"))
    proba = classifier.predict_proba(np.array([pixel_vec(img)], dtype = "float32"))
    return symbol_label_list[pred_index[0]]

def detect_cnn_proba(img, classifier):
    proba = classifier.predict_proba(np.array([pixel_vec(img)], dtype = "float32"))
    return proba

def load_cnn_classifier():
    return skflow.TensorFlowEstimator.restore(cnn_model_path)