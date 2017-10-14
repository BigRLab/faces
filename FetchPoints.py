import requests
import base64
import json
import itertools
from multiprocessing.dummy import Pool as ThreadPool
from pymongo import MongoClient
import os
import configparser


# -------------------------------
def main():
    parser = configparser.ConfigParser()
    parser.read('faces.cfg')

    # Get all the images - each image name is unique
    mypath = parser.get("image_path", "path")
    full_image_paths = []
    image_names = []
    for (dirpath, dirnames, filenames) in os.walk(mypath):
        for filename in filenames:
            full_image_paths.append(os.path.join(dirpath, filename))
            image_names.append(filename)

    # Init MongoDB connection
    conn = parser.get('mongodb', 'conn')
    client = MongoClient(conn)
    db = client.faces_db
    collection = db.raw_image_points

    # Get all image attributes - namely male or female
    malefemale_dict = {}
    with open(parser.get("image_path", "malefemale")) as f:
        for line in f:
            (key, val) = line.split()
            malefemale_dict[key] = val

    pool = ThreadPool(3)
    pool.starmap(process_image, zip(full_image_paths, image_names, itertools.repeat(collection),
                                    itertools.repeat(malefemale_dict), itertools.repeat(parser)))
    pool.close()
    pool.join()

    return


# -------------------------------
def process_image(image_path, image_name, db_collection, malefemale_dict, parser):
    # No attribute available - exit as we cant train using this image
    if image_name not in malefemale_dict:
        return

    cursor = db_collection.find({"file_name": image_name})
    if cursor.count() > 0:
        print("Image already processed - skipping {}".format(image_name))
        return

    # Private key for API
    headers = {
        "x-api-key": parser.get("trueface", "key"),
        "Content-Type": "application/json",
    }

    url = "https://api.trueface.ai/v1/facedetect?rawlandmarks=true"

    # Convert binary to bytes
    base64_bytes = base64.b64encode(open(image_path, 'rb').read())
    base64_string = base64_bytes.decode("utf-8")

    data = {
        'img': base64_string
    }

    print("Request Raw Image Data for {}".format(image_path))
    r = requests.post(url, data=json.dumps(data), headers=headers, timeout=None)
    print("Recieved Raw Image Data for {}".format(image_path))

    parsed_json = r.json()
    print(parsed_json)

    key = "success"
    if key in parsed_json and parsed_json[key] is True:
        parsed_json['file_name'] = image_name
        parsed_json['male_female'] = malefemale_dict[image_name]

        result = db_collection.insert_one(parsed_json)
        print("MongoDB ID: {}".format(result.inserted_id))

    return


# -------------------------------
if __name__ == "__main__":
    main()
