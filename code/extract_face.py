import vgg

from os import listdir
from os.path import isdir
from PIL import Image
from matplotlib import pyplot
from numpy import savez_compressed
from numpy import asarray
from mtcnn.mtcnn import MTCNN
import cv2
import numpy as np

import sklearn.model_selection
import tensorflow as tf

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from keras.models import Model
from pickle import dump

import torch
import torchvision.models as models
import torch.nn as nn
from PIL import Image
from torchvision import transforms
model = vgg.vgg_face_dag('vgg_face_dag.pth')
# model.classifier = nn.Sequential(*[model.classifier[i] for i in range(4)])
# print(model.classifier)

# Image size
IMAGE_SIZE = (224, 224)

# Load all labels
dictionary_labels = {}
with open("image/labels.txt") as f:
    for line in f:
       (key, val) = line.split()
       dictionary_labels[key] = val
       
total_image = len(dictionary_labels)

#data_path = 'image/img_align_celeba'
#data_path = 'image/test'
data_path = 'image/celeb'

# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory):
    count_progress = 0
    X, y = list(), list()
    # enumerate folders, on per class
    for img in listdir(directory):
        count_progress += 1
        print(count_progress)
        # path
        path = directory + '/' + img
        # load all faces in the subdirectory
        face = load_face(path)
        
        if len(face) == 0:
            continue
        
        # create labels
        label = dictionary_labels[img]
        # store
        X.append(face)
        y.append(label)
        # Print progress
        curr_percent = count_progress / total_image * 100
        if curr_percent % 10 == 0:
            print('Progress:', count_progress)
        
    return asarray(X), asarray(y)

# load images and extract faces for all images in a directory
def load_face(path):
    face = pytorch_vgg(path)
    return face.cpu().numpy()

# extract a single face from a given photograph
def pytorch_vgg(filename):
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        return model(input_batch)
    
    
X, y = load_dataset(data_path)
print('X shape:', X[0].shape)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, test_size=0.10, random_state=42)

# save arrays to one file in compressed format
savez_compressed('5-celebrity-faces-dataset.npz', X_train, y_train, X_test, y_test)