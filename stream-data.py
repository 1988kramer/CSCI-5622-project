# Andrew Kramer
# stream-data.py

# arranges data into a stream of examples
# examples have no defined beginning and end
# examples are randomly up or downsampled to have length of 0.5 to 2s

# currently getting pretty poor accuracy
# could be due to the difference between training and testing methods
# try altering the training data so it better reflects the test case

import numpy as np
from numpy.random import randint
from sklearn import preprocessing
from sklearn.svm import SVC
from scipy.signal import resample
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# loads sensor data and labels from the specified files
# shuffles examples and splits into test and training sets
# returns test and training examples and labels
def load_data(xfile, yfile, split):
	x_data = np.load(xfile) # load ordered x values
	y_data = np.load(yfile) # load ordered y values
	m = y_data.size

	#shuffle x and y values
	indices = np.array(range(0, m))
	np.random.shuffle(indices) 
	shuf_x = np.zeros(x_data.shape)
	shuf_y = np.zeros(m)
	for i in range(0,m):
		shuf_x[i, :, :] = x_data[indices[i], :, :] 
		shuf_y[i] = y_data[indices[i]] 
	
	# split into testing and training sets
	split_index = int(split * m)
	train_x = shuf_x[0:split_index]
	test_x = shuf_x[split_index:m]
	train_y = shuf_y[0:split_index]
	test_y = shuf_y[split_index:m]
	return train_x, train_y, test_x, test_y


# defines and trains an SVM on the given training data and labels
# returns the trained SVM
def train_model(train_x, train_y):
	# get mean for each training sample
	train_x = np.mean(train_x, axis=1)

	# normalize training data
	train_x = preprocessing.scale(train_x, axis=1)

	# define and train classifier
	svm = SVC(kernel='poly', degree=3, C=100, decision_function_shape='ovo')
	svm.fit(train_x, train_y)
	return svm

def create_stream(test_x, test_y):
	min_samples = 5
	max_samples = 20
	sample_length = randint(min_samples, max_samples)
	stream_x = resample(test_x[0], sample_length, axis=0)
	stream_y = np.full((sample_length, 1), test_y[0])
	for i in range(1, np.size(test_x, 0)):
		sample_length = randint(min_samples, max_samples)
		next_x = resample(test_x[i], sample_length, axis=0)
		next_y = np.full((sample_length, 1), test_y[i])
		stream_x = np.append(stream_x, next_x, axis=0)
		stream_y = np.append(stream_y, next_y, axis=0)
	return stream_x, stream_y

def classify_stream(stream_x, svm, win_sz=4):
	str_len = np.size(stream_x,0)
	stream_pred = np.zeros((str_len,1))
	up = int(np.ceil(win_sz / 2))
	dn = int(np.floor(win_sz / 2))
	for i in range(dn, str_len - dn):
		sample = np.mean(stream_x[i-dn:i+up, :], axis=0)
		sample = preprocessing.scale(sample)
		next_pred = svm.predict(np.reshape(sample, (1,-1)))
		stream_pred[i] = next_pred

	#stream_pred[str_len-win_sz:str_len] = stream_pred[str_len-win_sz]
	return stream_pred

def get_accuracy(stream_pred, stream_y):
	# naive implementation: calculate hamming distance and divide by length
	hamming = 0
	for (pred, ac) in zip(stream_pred, stream_y):
		if (pred != ac):
			hamming += 1;
	acc = 1 - (hamming / np.size(stream_pred,0))

	# plot classifications
	plt.plot(stream_pred[0:200])
	plt.plot(stream_y[0:200])
	plt.ylabel('predicted label')
	#plt.show()
	return acc

if __name__ == "__main__":
	xfile = "x_data.npy"
	yfile = "y_data.npy"
	print("loading data")
	train_x, train_y, test_x, test_y = load_data(xfile, yfile, 0.8)
	print("creating data streams")
	stream_x, stream_y = create_stream(test_x, test_y)
	print("training model")
	svm = train_model(train_x, train_y)
	print("predicting from data stream")
	stream_pred = classify_stream(stream_x, svm, 10)
	print("calculating accuracy")
	acc = get_accuracy(stream_pred, stream_y)
	print("accuracy: ", acc)
