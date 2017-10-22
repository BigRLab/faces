import configparser
import tensorflow as tf
import requests
import base64
import json
import NormalisePoints

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

    sess = tf.InteractiveSession()
    saver = tf.train.Saver(max_to_keep=None)

    saver.restore(sess, parser.get("tensor_model", "model_path"))

    # Import data
    binary_array = [ import_data(parser) ]

    # feed_dict is like named arguments for the accuracy function
    prediction = sess.run(y, feed_dict={x: binary_array})
    translate_prediction(prediction[0])

    return


def translate_prediction(pred):
    print("pred {}".format(pred))

    if pred[ 0 ] > pred[ 1 ]:
        print("Image is male")
    else:
        print("Image is female")

    return


def import_data(parser):
    face_path = parser.get("image_path", "test_image")
    print("Retrieving new face {}".format(face_path))

    normalised_width = parser.getint("normalised_size", "width")
    normalised_height = parser.getint("normalised_size", "height")

    # Private key for API
    headers = {
        "x-api-key": parser.get("trueface", "key"),
        "Content-Type": "application/json",
    }

    url = "https://api.trueface.ai/v1/facedetect?rawlandmarks=true"

    # Convert binary to bytes
    base64_bytes = base64.b64encode(open(face_path, 'rb').read())
    base64_string = base64_bytes.decode("utf-8")

    data = {
        'img': base64_string
    }

    print("Request Raw Image Data for {}".format(face_path))
    r = requests.post(url, data=json.dumps(data), headers=headers, timeout=None)
    print("Recieved Raw Image Data for {}".format(face_path))

    parsed_json = r.json()
    print(parsed_json)

    # flatten and return all points, we dont care about which part of the face they relate to for now
    face_landmarks = NormalisePoints.fetch_all_points(parsed_json["faces"][0]["landmarks"])

    print("face_landmarks: {}".format(face_landmarks))

    # boundary of the face within the photo
    face_boundary = parsed_json["faces"][0]["bounding_box"]

    # remove offset from top left corner
    top_left = NormalisePoints.move_image_top_left(face_landmarks, face_boundary)

    print("face_boundary: {}".format(face_boundary))
    print("top_left: {}".format(top_left))

    normalised_landmarks = NormalisePoints.normalise_landmarks(top_left, face_boundary, normalised_width, normalised_height)

    # Convert to binary true false array to represent each point
    binary_landmarks = NormalisePoints.convert_binary_landmarks(normalised_landmarks, normalised_width, normalised_height)

    NormalisePoints.print_ascii_face(binary_landmarks, normalised_width)

    return binary_landmarks


if __name__ == '__main__':
    tf.app.run(main=main, argv=[])
