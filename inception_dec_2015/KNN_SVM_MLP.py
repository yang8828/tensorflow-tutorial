# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
import glob
from skimage import io
import numpy as np
import imutils
import cv2
import os

def image_to_feature_vector(image, size=(128, 128)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
	return cv2.resize(image, size).flatten()

def extract_color_histogram(image, bins=(32, 32, 32)):
	# extract a 3D color histogram from the HSV color space using
	# the supplied number of `bins` per channel
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,[0, 180, 0, 256, 0, 256])

	# handle normalizing the histogram if we are using OpenCV 2.4.X
	if imutils.is_cv2():
		hist = cv2.normalize(hist)

	# otherwise, perform "in place" normalization in OpenCV 3
	else:
		cv2.normalize(hist, hist)

	# return the flattened histogram as the feature vector
	return hist.flatten()


print("handling images...")
path='/home/yangqr8828/Desktop/Test_DL/data/etl_image/'

rawImages = []
features = []
labels = []

cate=[path+x for x in os.listdir(path) if os.path.isdir(path+x)]
for idx,folder in enumerate(cate):
        for im in glob.glob(folder+'/*.jpg'):
            	img=io.imread(im)
            	pixels = image_to_feature_vector(img)
		hist = extract_color_histogram(img)
            	rawImages.append(pixels)
		features.append(hist)
		labels.append(idx)


# show some information on the memory consumed by the raw images
# matrix and features matrix
rawImages = np.array(rawImages)
features = np.array(features)
labels = np.array(labels)


# partition the data into training and testing splits, using 85%
# of the data for training and the remaining 15% for testing
(trainRI, testRI, trainRL, testRL) = train_test_split(
	rawImages, labels, test_size=0.15, random_state=42)
(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
	features, labels, test_size=0.15, random_state=42)


for i in range(2,20):
	#neural network
	print("\n")
	print("evaluating raw pixel accuracy for %d..." %(20*i))
	model = MLPClassifier(hidden_layer_sizes=(20*i,), max_iter=1000, alpha=1e-4,
		              solver='sgd', tol=1e-4, random_state=1,
		              learning_rate_init=.1)
	model.fit(trainRI, trainRL)
	acc = model.score(testRI, testRL)
	print("neural network raw pixel accuracy: {:.2f}%".format(acc * 100))


	#neural network
	print("\n")
	print("evaluating histogram accuracy...")
	model = MLPClassifier(hidden_layer_sizes=(20*i,), max_iter=1000, alpha=1e-4,
		              solver='sgd', tol=1e-4, random_state=1,
		              learning_rate_init=.1)
	model.fit(trainFeat, trainLabels)
	acc = model.score(testFeat, testLabels)
	print("neural network histogram accuracy : {:.2f}%".format(acc * 100))
'''
# k-NN
print("\n")
print("evaluating raw pixel accuracy...")
model = KNeighborsClassifier(n_neighbors=i)
model.fit(trainRI, trainRL)
acc = model.score(testRI, testRL)
print("k-NN classifier: k=%d" %i)
print("raw pixel accuracy: {:.2f}%".format(acc * 100))

# k-NN
print("\n")
print("evaluating histogram accuracy...")
model = KNeighborsClassifier(n_neighbors=i)
model.fit(trainFeat, trainLabels)
acc = model.score(testFeat, testLabels)
print(" k-NN classifier: k=%d" % i)
print(" histogram accuracy: {:.2f}%".format(acc * 100))
#SVC
print("\n")
print("evaluating raw pixel accuracy...")
model = SVC(max_iter=1000,class_weight='balanced')
model.fit(trainRI, trainRL)
acc = model.score(testRI, testRL)
print("SVM-SVC raw pixel accuracy: {:.2f}%".format(acc * 100))

#SVC
print("\n")
print("evaluating histogram accuracy...")
model = SVC(max_iter=1000,class_weight='balanced')
model.fit(trainFeat, trainLabels)
acc = model.score(testFeat, testLabels)
print("SVM-SVC histogram accuracy: {:.2f}%".format(acc * 100))
'''
