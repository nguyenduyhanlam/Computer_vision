# develop a classifier for the 5 Celebrity Faces Dataset
import numpy as np
from numpy import load
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from sklearn.decomposition import PCA

# load dataset
data = load('5-celebrity-faces-dataset.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Dataset: train=%d, test=%d' % (trainX.shape[0], testX.shape[0]))
print(trainX.shape)

PCA_COMPONENT = 400
pca = PCA(n_components=PCA_COMPONENT)

trainX = np.reshape(trainX, (len(trainX),4096))
testX = np.reshape(testX, (len(testX),4096))

trainX = pca.fit_transform(trainX)
testX = pca.fit_transform(testX)

# normalize input vectors
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)

# label encode targets
#out_encoder = LabelEncoder()
#out_encoder.fit(trainy)
#trainy = out_encoder.transform(trainy)
#testy = out_encoder.transform(testy)

# fit model
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainy)
# predict
yhat_train = model.predict(trainX)
yhat_test = model.predict(testX)
# score
score_train = accuracy_score(trainy, yhat_train)
score_test = accuracy_score(testy, yhat_test)
# summarize
print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))