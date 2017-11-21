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

import keras
from keras.models import Sequential
from keras.layers import Dense 
from keras.layers import Activation
from keras.utils import to_categorical
from keras import backend as K

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
def train_svm(train_x, train_y, win_sz):

	parsed_x, parsed_y = parse_stream(train_x, train_y, win_sz)

	# define and train classifier
	svm = SVC(kernel='poly', degree=3, C=100, decision_function_shape='ovo')
	svm.fit(parsed_x, np.ravel(parsed_y))
	return svm

def train_neural_net(train_x, train_y, win_sz):
	parsed_x, parsed_y = parse_stream(train_x, train_y, win_sz)
	cat_y = to_categorical(parsed_y, num_classes=5)

	model = Sequential()
	model.add(Dense(512, input_dim=16, activation='relu'))
	model.add(Dense(512, activation='relu'))
	model.add(Dense(512, activation='relu'))
	model.add(Dense(512, activation='relu'))
	model.add(Dense(5, activation='softmax'))
	model.compile(loss='categorical_crossentropy', 
				  optimizer='sgd', 
				  metrics=['accuracy'])

	model.fit(parsed_x, cat_y, epochs=35, batch_size=32)

	return model

def test_neural_net(neural_net, test_x, test_y, win_sz):
	parsed_x, parsed_y = parse_stream(test_x, test_y, win_sz)
	cat_y = to_categorical(parsed_y, num_classes=5)

	acc = neural_net.evaluate(parsed_x, cat_y)
	print("neural net accuracy: ", acc)

def parse_stream(stream_x, stream_y, win_sz):
	str_len = np.size(stream_x, 0)
	parsed_x = np.zeros((str_len - win_sz + 1, 16))
	parsed_y = np.zeros((str_len - win_sz + 1, 1))
	up = int(np.ceil(win_sz / 2))
	dn = int(np.floor(win_sz / 2))
	for i in range(dn, str_len - up):
		sample = np.mean(stream_x[i-dn:i+up, :], axis=0)
		sample = preprocessing.scale(sample)
		parsed_x[i - dn] = sample
		parsed_y[i - dn] = stream_y[i]
	return parsed_x, parsed_y

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

def classify_stream_svm(stream_x, stream_y, svm, win_sz):
	parsed_x, parsed_y = parse_stream(stream_x, stream_y, win_sz)
	pred_y = svm.predict(parsed_x)
	get_accuracy(pred_y, np.ravel(parsed_y))


def get_accuracy(y_pred, y_test):

	print(classification_report(y_pred, y_test))

	# plot classifications
	plt.plot(y_pred[0:200])
	plt.plot(y_test[0:200])
	plt.ylabel('predicted label')
	#plt.show()
	return acc

if __name__ == "__main__":
	xfile = "x_data.npy"
	yfile = "y_data.npy"
	window_size = 5
	print("loading data")
	train_x, train_y, test_x, test_y = load_data(xfile, yfile, 0.8)
	print("creating data streams")
	train_x_str, train_y_str = create_stream(train_x, train_y)
	test_x_str, test_y_str = create_stream(test_x, test_y)
	print("training model")
	svm = train_svm(train_x_str, train_y_str, window_size)
	#neural_net = train_neural_net(train_x_str, train_y_str, window_size)
	print("predicting from data stream")
	stream_pred = classify_stream_svm(test_x_str, test_y_str, svm, window_size)
	#test_neural_net(neural_net, test_x_str, test_y_str, window_size)
