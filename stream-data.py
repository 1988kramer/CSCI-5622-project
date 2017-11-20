# Andrew Kramer
# stream-data.py

# arranges data into a stream of examples
# examples have no defined beginning and end
# examples are randomly up or downsampled to have length of 0.5 to 2s

import numpy as np
from np.random import randint
from sklearn import preprocessing
from sklearn.svm import SVC
from scipy.signal import resample
from sklearn.metrics import classification_report

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
	train_x = preprocessing.scale(train_x)

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

def classify_stream(stream_x, svm):
	# classify examples in data stream
	return stream_pred

def get_accuracy(stream_pred, stream_y):