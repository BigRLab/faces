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
    y_predict = tf.matmul(x, W) + b

    # Define loss and optimizer
    y_truth = tf.placeholder(tf.float32, [None, 2])

    saver = tf.train.Saver(max_to_keep=None)

    # Import data
    eval_data = import_data(parser)
    pivoted_data = pivot_data(eval_data)

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_truth, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    image_binary_arrays = pivoted_data[0]
    male_female_labels = pivoted_data[1]

    with tf.Session() as sess:
        saver.restore(sess, parser.get("tensor_model", "model_path"))

        # feed_dict is like named arguments for the accuracy function, our function requires x and y_truth as arguments
        accuracy_percentage = sess.run(accuracy, feed_dict={x: image_binary_arrays, y_truth: male_female_labels})
        print("Model accuracy: {}".format(accuracy_percentage))

    return


def import_data(parser):
    print("Retrieving eval dataset")

    # Init MongoDB connection
    conn = parser.get('mongodb', 'training_conn')
    client = MongoClient(conn)
    db = client.faces_db
    eval_collection = db.eval_collection

    image_data = []
    male_count = 0

    pipeline = [
        {"$match": {"male_female": "-1"}},
        {"$group": {"_id": "$male_female", "count": {"$sum": 1}}},
    ]
    cursor = eval_collection.aggregate(pipeline)
    for entry in cursor:
        female_images = entry["count"]
        print("Number of female images: {}".format(female_images))

    for image_json in eval_collection.find():
        male_female = [0, 1]
        if int(image_json["male_female"]) > 0:
            if male_count >= female_images: # there are many more male images, we dont use them to ensure no bias result
                continue

            male_female = [1, 0]
            male_count += 1

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
