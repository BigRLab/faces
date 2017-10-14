import requests
import base64
import json
import itertools
from multiprocessing.dummy import Pool as ThreadPool 
from pymongo import MongoClient
import os

#-------------------------------
def main():
	# Get all the images - each image name is unique
	mypath = 'D:\LFW\lfw_funneled\Aaron_Eckhart'
	fullImagePaths = []
	imageNames = []
	for (dirpath, dirnames, filenames) in os.walk(mypath):
		for filename in filenames:
			fullImagePaths.append(os.path.join(dirpath, filename))
			imageNames.append(filename)

	# Init MongoDB connection
	client = MongoClient('mongodb://fetcher:fetcherpass@ds119395.mlab.com:19395/faces_db')
	db = client.faces_db
	collection = db.raw_image_points

	# Get all image attributes - namely male or female
	malefemaleDict = {}
	with open("D:\LFW\MaleFemale.txt") as f:
		for line in f:
			(key, val) = line.split()
			malefemaleDict[key] = val

	pool = ThreadPool(3)
	results = pool.starmap(processImage, zip(fullImagePaths, imageNames, itertools.repeat(collection), itertools.repeat(malefemaleDict)))
	pool.close()
	pool.join()

	return

#-------------------------------
def processImage(imagePath, imageName, mongoCollection, malefemaleDict):
	# No attribute available - exit as we cant train using this image
	if imageName not in malefemaleDict:
		return

	cursor = mongoCollection.find({"file_name": imageName})
	if cursor.count() > 0:
		print( "Image already processed - skipping {}".format( imageName ) )
		return

	# Private key for API
	headers = {
		"x-api-key":"KVz3VVGVA19hOrEFlvQoqao29z3qdht66IiHplK8",
		"Content-Type":"application/json",
	}

	url = "https://api.trueface.ai/v1/facedetect?rawlandmarks=true"

	# Convert binary to bytes
	base64_bytes = base64.b64encode(open(imagePath, 'rb').read())
	base64_string = base64_bytes.decode("utf-8")

	data = {
		'img':base64_string
	}

	print( "Request Raw Image Data for {}".format( imagePath ) )
	r = requests.post(url,data=json.dumps(data),headers=headers, timeout=None)
	print( "Recieved Raw Image Data for {}".format( imagePath ) )

	parsed_json = r.json()
	print( parsed_json )

	key = "success"
	if key in parsed_json and parsed_json[key] == True:
		parsed_json['file_name']=imageName
		parsed_json['male_female']=malefemaleDict[imageName]

		result = mongoCollection.insert_one(parsed_json)
		print( "MongoDB ID: {}".format( result.inserted_id ) )
	
	return

#-------------------------------
if __name__ == "__main__":
    main()
