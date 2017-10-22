import configparser
from pymongo import MongoClient
import tensorflow as tf

FLAGS = None


def main(_):
    parser = configparser.ConfigParser()
    parser.read('faces.cfg')

    normalised_width = parser.getint("normalised_size", "width")
    normalised_height = parser.getint("normalised_size", "height")

    # Create the model
    binary_array_size = normalised_width * normalised_height
    # Create the model
    x = tf.placeholder(tf.float32, [None, binary_array_size])
    W = tf.Variable(tf.zeros([binary_array_size, 2]))
    b = tf.Variable(tf.zeros([2]))
    y = tf.matmul(x, W) + b

    # Define loss and optimizer
    y_truth = tf.placeholder(tf.float32, [None, 2])

    sess = tf.InteractiveSession()
    saver = tf.train.Saver(max_to_keep=None)

    saver.restore(sess, parser.get("tensor_model", "model_path"))

    # Import data
    eval_data = import_data(parser)
    pivoted_data = pivot_data(eval_data)

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_truth, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    images = pivoted_data[0]
    labels = pivoted_data[1]
    print(sess.run(accuracy, feed_dict={x: images, y_truth: labels}))


def import_data(parser):
    print("Retrieving eval dataset")

    # Init MongoDB connection
    conn = parser.get('mongodb', 'training_conn')
    client = MongoClient(conn)
    db = client.faces_db
    eval_collection = db.eval_collection

    image_data = []

    for image_json in eval_collection.find():
        male_female = [0, 1]
        if int(image_json["male_female"]) > 0:
            male_female = [1, 0]

        data = ( image_json["binary_landmarks"], male_female )
        image_data.append(data)

    print("Eval dataset size: {}".format(len(image_data)))
    return image_data


def pivot_data(dataset):
    binary_array = []
    male_female_array = []
    for t in dataset:
        binary_array.append(t[0])
        male_female_array.append(t[1])

    return(binary_array, male_female_array)

if __name__ == '__main__':
    tf.app.run(main=main, argv=[])
