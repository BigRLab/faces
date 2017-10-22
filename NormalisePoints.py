from pymongo import MongoClient
import math
import configparser


# -------------------------------
def main():
    parser = configparser.ConfigParser()
    parser.read('faces.cfg')

    # Init MongoDB connection
    conn = parser.get('mongodb', 'conn')
    client = MongoClient(conn)
    db = client.faces_db
    raw_collection = db.raw_image_points
    normalised_collection = db.normalised_collection

    # Go through all the raw images
    for image_json in raw_collection.find():
        json = process_image(image_json, normalised_collection,
                      parser.getint("normalised_size", "width"), parser.getint("normalised_size", "height"))

    return


# -------------------------------
def process_image(image_json, normalised_collection, normalised_width, normalised_height):
    # In case we have already done this one
    cursor = normalised_collection.find({"file_name": image_json["file_name"]})
    if cursor.count() > 0:
        print("Image already processed - skipping {}".format(image_json["file_name"]))
        return
    else:
        print("Processing {}".format(image_json["file_name"]))

    # for debugging Romano_Prodi_0005
    #if image_json["file_name"] != "Bill_Simon_0015.jpg":
    #    return

    if len( image_json["faces"] ) > 1:
        print("Skipping image {} due to multiple faces".format(image_json["file_name"]))
        return

    # flatten and return all points, we dont care about which part of the face they relate to for now
    face_landmarks = fetch_all_points(image_json["faces"][0]["landmarks"])

    print("face_landmarks: {}".format(face_landmarks))

    # boundary of the face within the photo
    face_boundary = image_json["faces"][0]["bounding_box"]

    # remove offset from top left corner
    top_left = move_image_top_left(face_landmarks, face_boundary)

    print("face_boundary: {}".format(face_boundary))
    print("top_left: {}".format(top_left))

    normalised_landmarks = normalise_landmarks(top_left, face_boundary, normalised_width, normalised_height)

    # Convert to binary true false array to represent each point
    binary_landmarks = convert_binary_landmarks(normalised_landmarks, normalised_width, normalised_height)

    print_ascii_face(binary_landmarks, normalised_width)

    json = {"file_name": image_json["file_name"], "normalised_landmarks": normalised_landmarks,
            "binary_landmarks": binary_landmarks, "male_female": image_json["male_female"]}

    result = normalised_collection.insert_one(json)
    print( "MongoDB ID: {}".format( result.inserted_id ) )

    return json


# -------------------------------
def print_ascii_face(binary_landmarks, width):

    for i in range(len(binary_landmarks)):
        if binary_landmarks[i] == 1:
            print("X", end="")
        else:
            print(".", end="")

        if (i + 1) % width == 0:
            print("")

    return

# -------------------------------
def move_image_top_left(face_landmarks, face_boundary):
    centered_marks = []

    for point in face_landmarks:
        x = math.floor(point[0] - face_boundary[0])
        y = math.floor(point[1] - face_boundary[1])
        centered_marks.append([x, y])

    return centered_marks


# -------------------------------
def convert_binary_landmarks(normalised_landmarks, normalised_width, normalised_height):
    binary_array = [0] * normalised_width * normalised_height

    print("Size {}".format(len(binary_array)))

    for mark in normalised_landmarks:
        # print( "mark[ 0 ] {}, mark[ 1 ] {}, index {}".format( mark[ 0 ], mark[ 1 ], mark[ 1 ] * normalised_width + mark[ 0 ] ) )
        binary_array[mark[1] * normalised_width + mark[0]] = 1

    return binary_array


# -------------------------------
def normalise_landmarks(face_landmarks, face_boundary, normalised_width, normalised_height):
    normalised_landmarks = []

    face_width = face_boundary[2] - face_boundary[0]
    face_height = face_boundary[3] - face_boundary[1]

    print("face_width {}, face_height {}".format(face_width, face_height))

    width_ratio = normalised_width / face_width
    height_ratio = normalised_height / face_height

    print("width_ratio {}, height_ratio {}".format(width_ratio, height_ratio))

    smaller_ratio = width_ratio
    height_adjust = (normalised_height - (face_height * smaller_ratio)) / 2
    width_adjust = 0
    if height_ratio < width_ratio:
        smaller_ratio = height_ratio
        height_adjust = 0
        width_adjust = (normalised_width - (face_width * smaller_ratio)) / 2

    print("smaller_ratio {}, width_adjust {}, height_adjust {}".format(smaller_ratio, width_adjust, height_adjust))

    for point in face_landmarks:
        x = min(normalised_width - 1, math.ceil(point[0] * smaller_ratio + width_adjust))
        y = min(normalised_height - 1, math.ceil(point[1] * smaller_ratio + height_adjust))
        print([x, y], end=" ")
        normalised_landmarks.append([x, y])

    return normalised_landmarks


# -------------------------------
def fetch_all_points(raw_landmarks):
    all_points = []

    # flatten and return all points, we dont care about which part of the face they relate to for now
    element = raw_landmarks[0][0]
    all_points.extend(element["nose_bridge"])
    all_points.extend(element["left_eye"])
    all_points.extend(element["nose_tip"])
    all_points.extend(element["chin"])
    all_points.extend(element["right_eye"])
    all_points.extend(element["left_eyebrow"])
    all_points.extend(element["bottom_lip"])
    all_points.extend(element["right_eyebrow"])
    all_points.extend(element["top_lip"])

    return all_points


# -------------------------------
if __name__ == "__main__":
    main()
