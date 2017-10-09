import requests
import base64
import json
import itertools
from multiprocessing.dummy import Pool as ThreadPool 
from pymongo import MongoClient

#-------------------------------
def main():

	images = [
		'C:/Users/Zhujin/Desktop/IMG_8040.JPG',
		'C:/Users/Zhujin/Desktop/card.JPG',
		'C:/Users/Zhujin/Desktop/kingston-university-dd25d4b-.JPG',
		'C:/Users/Zhujin/Desktop/IMG_8040.JPG',
		'C:/Users/Zhujin/Desktop/card.JPG',
		'C:/Users/Zhujin/Desktop/kingston-university-dd25d4b-.JPG',
		'C:/Users/Zhujin/Desktop/IMG_8040.JPG',
		'C:/Users/Zhujin/Desktop/card.JPG',
		'C:/Users/Zhujin/Desktop/kingston-university-dd25d4b-.JPG',
		'C:/Users/Zhujin/Desktop/IMG_8040.JPG',
		'C:/Users/Zhujin/Desktop/card.JPG',
		'C:/Users/Zhujin/Desktop/kingston-university-dd25d4b-.JPG',
		'C:/Users/Zhujin/Desktop/IMG_8040.JPG',
		'C:/Users/Zhujin/Desktop/card.JPG',
		'C:/Users/Zhujin/Desktop/kingston-university-dd25d4b-.JPG',
		'C:/Users/Zhujin/Desktop/IMG_8040.JPG',
		'C:/Users/Zhujin/Desktop/card.JPG',
		'C:/Users/Zhujin/Desktop/kingston-university-dd25d4b-.JPG',
	]

	client = MongoClient('mongodb://localhost:27017/')
	db = client.faces_db
	collection = db.raw_image_points

	pool = ThreadPool(20)
	results = pool.starmap(processImage, zip(images, itertools.repeat(collection)))
	pool.close()
	pool.join()

	return

#-------------------------------
def processImage(imagePath, mongoCollection):
	headers = {
		"x-api-key":"KVz3VVGVA19hOrEFlvQoqao29z3qdht66IiHplK8",
		"Content-Type":"application/json",
	}

	url = "https://api.trueface.ai/v1/facedetect?rawlandmarks=true"

	base64_bytes = base64.b64encode(open(imagePath, 'rb').read())
	base64_string = base64_bytes.decode("utf-8")

	data = {
		'img':base64_string
	}

	r = requests.post(url,data=json.dumps(data),headers=headers, timeout=None)

	result = mongoCollection.insert_one(r.json())

	print( result.inserted_id )
	print( r.json() )
	
	return

#-------------------------------
if __name__ == "__main__":
    main()
