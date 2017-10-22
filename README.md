# faces
Image Machine Learning using TensorFlow by looking at facial landmarks provided by trueface.ai - backed by MongoDB mlab service

FetchPoints.py - this is used to loop through all images (10k+), call trueface.ai landmarks API and persist to MongoDB
NormalisePoints.py - this is used to loop through all results and normalise the facial landmarks into a standard 40x40 pixel box, this normalisation is required as input for our ML training model requires standard size
SplitDataSets.py - this is used to split the dataset into training vs eval data sets
Train Model.py - takes the training dataset and uses TensorFlow to train the model, this is using a very basic model as I havent had time to explore more advance ML techniques yet
Eval Model.py - evaluates how well the model performs against our training set
Eval New Face.py - this is a end product if you like, which uses the ML model and outputs its prediction against new image you supply

faces.cfg is required for all the scripts, this is not uploaded to protect private keys
