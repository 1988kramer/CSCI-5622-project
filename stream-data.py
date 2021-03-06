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
from sklearn import preprocessing, linear_model
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

# loads data that has already been shuffled and organized into streams
def load_stream_data(xfile, yfile, split):
	x_stream = np.load(xfile)
	y_stream = np.load(yfile)
	m = y_stream.size
	split_index = int(split * m)
	train_x = x_stream[0:split_index]
	test_x  = x_stream[split_index:m]
	train_y = y_stream[0:split_index]
	test_y  = y_stream[split_index:m]
	return train_x, train_y, test_x, test_y


# defines and trains an SVM on the given training data and labels
# returns the trained SVM
def train_svm(train_x, train_y, win_sz):

	parsed_x, parsed_y = parse_stream(train_x, train_y, win_sz)

	# define and train classifier
	svm = SVC(kernel='poly', degree=3, C=1000, decision_function_shape='ovo')
	svm.fit(parsed_x, np.ravel(parsed_y))
	return svm

def train_neural_net(train_x, train_y, win_sz, classes):
	parsed_x, parsed_y = parse_stream(train_x, train_y, win_sz)
	cat_y = to_categorical(parsed_y, num_classes=classes)

	model = Sequential()
	model.add(Dense(512, input_dim=16, activation='relu'))
	#model.add(Dense(512, activation='relu'))
	model.add(Dense(512, activation='relu'))
	model.add(Dense(512, activation='relu'))
	model.add(Dense(classes, activation='softmax'))
	model.compile(loss='categorical_crossentropy', 
				  optimizer='adam', 
				  metrics=['accuracy'])

	model.fit(parsed_x, cat_y, epochs=10, batch_size=16)

	return model

def test_neural_net(neural_net, test_x, test_y, win_sz, classes):
	parsed_x, parsed_y = parse_stream(test_x, test_y, win_sz)
	cat_y = to_categorical(parsed_y, num_classes=classes)

	acc = neural_net.evaluate(parsed_x, cat_y)
	pred_y = neural_net.predict(parsed_x)
	print("neural net accuracy: ", acc)
	return pred_y, cat_y

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

def train_logistic(train_x, train_y, win_sz):
	parsed_x, parsed_y = parse_stream(train_x, train_y, win_sz)
	logreg = linear_model.LogisticRegression(multi_class='multinomial', 
											 solver='saga', C=.01, 
											 max_iter=5000)

	logreg.fit(parsed_x, np.ravel(parsed_y))
	return logreg
	

def classify_logistic(test_x, test_y, logreg, win_sz):
	parsed_x, parsed_y = parse_stream(test_x, test_y, win_sz)
	pred = logreg.predict(parsed_x)
	print("Logistic regression results: ")
	print(classification_report(np.ravel(parsed_y), pred))

def get_accuracy(y_pred, y_test):

	print(classification_report(y_pred, y_test))

	# plot classifications
	plt.plot(y_pred[0:200])
	plt.plot(y_test[0:200])
	plt.ylabel('predicted label')
	#plt.show()

def from_categorical(cat):
	return np.argmax(cat, axis=1);


if __name__ == "__main__":
	xfile = "x_stream.npy"
	yfile = "y_stream.npy"
	window_size = 1
	classes = 5
	six_cat = False
	if (six_cat):
		train_x_str, train_y_str, test_x_str, test_y_str = load_stream_data(xfile, yfile, 0.8)
	else:
		train_x, train_y, test_x, test_y = load_data("x_data.npy", "y_data.npy", 0.8)
		train_x_str, train_y_str = create_stream(train_x, train_y)
		test_x_str, test_y_str = create_stream(test_x, test_y)
	#svm = train_svm(train_x_str, train_y_str, window_size)
	#print("svm results")
	#classify_stream_svm(test_x_str, test_y_str, svm, window_size)
	if (six_cat):
		classes = 6
	neural_net = train_neural_net(train_x_str, train_y_str, window_size, classes)
	print("neural net results")
	pred_y, test_y = test_neural_net(neural_net, test_x_str, test_y_str, window_size, classes)
	np.savetxt("nn_pred.csv", from_categorical(pred_y), delimiter = ",")
	np.savetxt("nn_test.csv", from_categorical(test_y), delimiter = ",")
	#get_accuracy(pred_y, test_y); 
	#logistic = train_logistic(train_x_str, train_y_str, window_size)
	#print("logistic results")
	#classify_logistic(test_x_str, test_y_str, logistic, window_size)
