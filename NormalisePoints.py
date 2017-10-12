from pymongo import MongoClient
import math

#-------------------------------
def main():
	# Init MongoDB connection
	client = MongoClient('mongodb://localhost:27017/')
	db = client.faces_db
	raw_collection = db.raw_image_points
	normalised_collection = db.normalised_collection

	# Go through all the raw images
	for image_json in raw_collection.find():
		processImage(image_json, normalised_collection, 100, 120)

	return

#-------------------------------
def processImage(image_json, normalised_collection, normalised_width, normalised_height):
	# In case we have already done this one
	cursor = normalised_collection.find({"file_name": image_json["file_name"]})
	if cursor.count() > 0:
		print( "Image already processed - skipping {}".format( image_json["file_name"] ) )
		return

	#for debugging
	if image_json["file_name"] != "Romano_Prodi_0005.jpg":
		return

	print( image_json )
	
	#if len( image_json["faces"][0]["landmarks"][0][0] ) > 1:
	#	return

	# flatten and return all points, we dont care about which part of the face they relate to for now
	face_landmarks = fetch_all_points( image_json["faces"][0]["landmarks"] )

	face_boundary = image_json["faces"][0]["bounding_box"]
	normalised_landmarks = normalise_landmarks( face_landmarks, face_boundary, normalised_width, normalised_height )

	json = {"file_name": image_json["file_name"],"normalised_landmarks":normalised_landmarks}

	#result = normalised_collection.insert_one(json)
	#print( "MongoDB ID: {}".format( result.inserted_id ) )
	
	print( json )

	return

#-------------------------------
def normalise_landmarks(face_landmarks, face_boundary, normalised_width, normalised_height):
	normalised_landmarks = []

	face_width = face_boundary[2] - face_boundary[0]
	face_height = face_boundary[3] - face_boundary[1]

	width_ratio = normalised_width / face_width
	height_ratio = normalised_height / face_height

	print( face_width )
	print( face_height )

	for point in face_landmarks:
		x = math.ceil(( point[0] - face_boundary[0] ) * width_ratio )
		y = math.ceil(( point[1] - face_boundary[1] ) * height_ratio )
		normalised_landmarks.append([ x, y ])

	return normalised_landmarks

#-------------------------------
def fetch_all_points( raw_landmarks ):
	all_points = []

	element = raw_landmarks[ 0 ][ 0 ]
	all_points.extend(element[ "nose_bridge" ])
	all_points.extend(element[ "left_eye" ])
	all_points.extend(element[ "nose_tip" ])
	all_points.extend(element[ "chin" ])
	all_points.extend(element[ "right_eye" ])
	all_points.extend(element[ "left_eyebrow" ])
	all_points.extend(element[ "bottom_lip" ])
	all_points.extend(element[ "right_eyebrow" ])
	all_points.extend(element[ "top_lip" ])

	return all_points

#-------------------------------
if __name__ == "__main__":
    main()
