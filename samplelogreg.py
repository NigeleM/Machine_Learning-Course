import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn import metrics

dataset = load_digits()
print("Number of samples: %d" % len(dataset.target))
features = dataset.data

print("feature Vectors: %s" % features)
labels = dataset.target
print("Label: %s " % labels)

trainIdx = np.random.rand(len(labels)) < 0.8
features_train = features[trainIdx]
labels_train = labels[trainIdx]
features_test = features[~trainIdx]
labels_test = labels[~trainIdx]

print("Number of training samples:" ,features_train.shape[0])
print("Number of test samples:" ,features_test.shape[0])
print("Feature vector dimensionality:" ,features_train.shape[1])

# logistic regression

from sklearn.linear_model import LogisticRegression
modeldata = LogisticRegression()
modeldata.fit(features_train,labels_train)
labelprediction = modeldata.predict(features_test)

print(metrics.classification_report(labels_test,labelprediction))
print(metrics.confusion_matrix(labels_test,labelprediction))
