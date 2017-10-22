import configparser
from pymongo import MongoClient
import random

import tensorflow as tf

FLAGS = None


def main(_):
    parser = configparser.ConfigParser()
    parser.read('faces.cfg')

    normalised_width = parser.getint("normalised_size", "width")
    normalised_height = parser.getint("normalised_size", "height")

    # Create the model
    binary_array_size = normalised_width * normalised_height

    x = tf.placeholder(tf.float32, [None, binary_array_size])
    W = tf.Variable(tf.zeros([binary_array_size, 2])) # [binary_array size, male_female]
    b = tf.Variable(tf.zeros([2]))
    y_predict = tf.matmul(x, W) + b

    # Define loss and optimizer
    y_truth = tf.placeholder(tf.float32, [None, 2])

    # The raw formulation of cross-entropy,
    #
    #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
    #                                 reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    # outputs of 'y', and then average across the batch.
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=y_predict,labels=y_truth))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    saver = tf.train.Saver(max_to_keep=None)

    # Import data
    training_data = import_data(parser)

    # Train
    for step in range(1000):
        print("Training step {}".format(step))
        batch_xs, batch_ys = get_random_sample(training_data, 100)
        sess.run(train_step, feed_dict={x: batch_xs, y_truth: batch_ys})

    saver.save(sess, 'D:\LFW\model.ckpt')


def import_data(parser):
    print("Retrieving training dataset")

    # Init MongoDB connection
    conn = parser.get('mongodb', 'training_conn')
    client = MongoClient(conn)
    db = client.faces_db
    training_collection = db.training_collection

    image_data = []

    for image_json in training_collection.find():
        male_female = [0, 1]
        if int(image_json["male_female"]) > 0:
            male_female = [1, 0]

        data = ( image_json["binary_landmarks"], male_female )
        image_data.append(data)

    print("Training dataset size: {}".format(len(image_data)))
    return image_data


def get_random_sample(dataset, sample_size):
    random_set = random.sample(dataset, sample_size)

    binary_array = []
    male_female_array = []
    for t in random_set:
        binary_array.append(t[0])
        male_female_array.append(t[1])

    return(binary_array, male_female_array)

if __name__ == '__main__':
    tf.app.run(main=main, argv=[])
