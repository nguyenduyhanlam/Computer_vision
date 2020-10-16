from os import listdir
from os.path import isdir
from PIL import Image
from matplotlib import pyplot
from numpy import savez_compressed
from numpy import asarray
from mtcnn.mtcnn import MTCNN

import sklearn

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from keras.models import Model
from pickle import dump

# Image size
IMAGE_SIZE = (224, 224)

# Load all labels
dictionary_labels = {}
with open("image/labels.txt") as f:
    for line in f:
       (key, val) = line.split()
       dictionary_labels[key] = val
       
total_image = len(dictionary_labels)

data_path = 'image/img_align_celeba'

# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory):
    count_progress = 0
    X, y = list(), list()
    # enumerate folders, on per class
    for img in listdir(directory):
        # path
        path = directory + '/' + img
        # load all faces in the subdirectory
        face = load_face(path)
        # create labels
        label = dictionary_labels[img]
        # store
        X.append(face)
        y.append(label)
        
        # Print progress
        count_progress += 1
        curr_percent = count_progress / total_image * 100
        if curr_percent % 10 == 0:
            print('Progress:', count_progress)
        
    return asarray(X), asarray(y)

# load images and extract faces for all images in a directory
def load_face(path):
    face = extract_face(path)
    return face

# extract a single face from a given photograph
def extract_face(filename, required_size=IMAGE_SIZE):
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    # bug fix
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    
    image = asarray(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    # load model
    model = VGG16()
    # remove the output layer
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    # get extracted features
    features = model.predict(image)
    #print(features.shape)
    #return face_array
    return features

X, y = load_dataset(data_path)
print('X shape:', X[0].shape)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, test_size=0.33, random_state=42)

# save arrays to one file in compressed format
savez_compressed('5-celebrity-faces-dataset.npz', X_train, y_train, X_test, y_test)