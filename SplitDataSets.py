from pymongo import MongoClient
import configparser
import random

# -------------------------------
def main():
    parser = configparser.ConfigParser()
    parser.read('faces.cfg')

    # Init MongoDB connection
    conn = parser.get('mongodb', 'conn')
    client = MongoClient(conn)
    db = client.faces_db
    normalised_collection = db.normalised_collection

    training_conn = parser.get('mongodb', 'training_conn')
    training_client = MongoClient(training_conn)
    training_db = training_client.faces_db
    training_collection = training_db.training_collection
    eval_collection = training_db.eval_collection

    print("Total sample size {}".format(normalised_collection.count()))
    print("Previous training size {}".format(training_collection.count()))
    print("Previous eval size {}".format(eval_collection.count()))

    training_db.training_collection.drop()
    training_db.eval_collection.drop()

    # Go through all the raw images
    for image_json in normalised_collection.find():
        if random.randint(0,9) >= 3:
            training_collection.insert_one(image_json)
        else:
            eval_collection.insert_one(image_json)

    print("New training size {}".format(training_collection.count()))
    print("New eval size {}".format(eval_collection.count()))

    return

# -------------------------------
if __name__ == "__main__":
    main()
