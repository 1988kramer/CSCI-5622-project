# Andrew Kramer
# test-classifiers.py

# tests two simple classifiers on examples of uniform length

# labels:
# 0 - fist
# 1 - down
# 2 - up
# 3 - left
# 4 - right

import numpy as np
from sklearn import linear_model, preprocessing
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from keras.models import Sequential
from keras.layers import Conv1D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import MaxPooling1D
from keras.layers import LSTM
from keras.layers import Dropout
from keras.utils import to_categorical

def load_data(xfile, yfile, split):
	x_data = np.load(xfile) # load ordered x values
	y_data = np.load(yfile) # load ordered y values
	m = y_data.size
	x_data = np.reshape(x_data, (m,-1)) # reshape to 2-D vector

	x_data = preprocessing.scale(x_data, axis=1) # normalize data

	#shuffle x and y values
	indices = np.array(range(0, m))
	np.random.shuffle(indices) 
	shuf_x = np.zeros(x_data.shape)
	shuf_y = np.zeros(m)
	for i in range(0,m):
		shuf_x[i, :] = x_data[indices[i], :] 
		shuf_y[i] = y_data[indices[i]] 
	
	# split into testing and training sets
	split_index = int(split * m)
	train_x = shuf_x[0:split_index]
	test_x = shuf_x[split_index:m]
	train_y = shuf_y[0:split_index]
	test_y = shuf_y[split_index:m]
	return train_x, train_y, test_x, test_y

def classifyLogistic(train_x, train_y, test_x, test_y):
	pca = PCA(n_components=100);
	pca.fit(train_x)

	logreg = linear_model.LogisticRegression(multi_class='multinomial', 
											 solver='saga', C=.01, 
											 max_iter=5000)

	pipe = Pipeline(steps=[('pca', pca), ('logistic', logreg)])

	pipe.fit(train_x, train_y)
	pred = pipe.predict(test_x)
	print("Logistic regression results: ")
	print(classification_report(test_y, pred))

def classifySVM(train_x, train_y, test_x, test_y):
	poly = SVC(kernel='poly', degree=3, C=1000, decision_function_shape='ovo')
	poly.fit(train_x, train_y)
	pred = poly.predict(test_x)
	print("Polynomial SVM results: ")
	print(classification_report(test_y, pred))

def classifyNN(train_x, train_y, test_x, test_y):
	act = 'relu'
	opt = 'Adadelta'
	loss = 'categorical_crossentropy'
	cat_train = to_categorical(train_y, num_classes=5)
	cat_test = to_categorical(test_y, num_classes=5)
	net = Sequential()
	#yields accuracy in the mid-high 70s
	net.add(Dense(16, activation=act, input_dim=train_x.shape[1]))
	net.add(Dense(64, activation=act))
	net.add(Dense(128, activation=act))
	net.add(Dropout(.2))
	net.add(Dense(256, activation=act))
	net.add(Dense(5, activation='softmax'))
	net.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
	net.fit(train_x, cat_train)
	print(net.evaluate(test_x, cat_test))

def classifyCNN(train_x, train_y, test_x, test_y):
	act = 'relu'
	opt = 'Adadelta'
	loss = 'categorical_crossentropy'
	train_tensor = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
	cat_train = to_categorical(train_y, num_classes=5)
	test_tensor = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))
	cat_test = to_categorical(test_y, num_classes=5)
	#this is the best topology I've found so far
	#tends to give low to mid 80s for test accuracy
	net =  Sequential()
	net.add(Conv1D(16, kernel_size=(3), input_shape=(train_x.shape[1], 1)))
	net.add(MaxPooling1D())
	net.add(Flatten())
	net.add(Dense(64, activation=act))
	net.add(Dense(5, activation='softmax'))
	net.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
	net.fit(train_tensor, cat_train)
	print(net.evaluate(test_tensor, cat_test))

def classifyRNN(train_x, train_y, test_x, test_y):
	opt = 'Adam'
	loss = 'categorical_crossentropy'
	train_tensor = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
	cat_train = to_categorical(train_y, num_classes=5)
	test_tensor = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))
	cat_test = to_categorical(test_y, num_classes=5)
	net = Sequential()
	#This gives about 32% accuracy at best,
	#and that's the best I've been able to find. We just don't
	#have enough data to get this better
	net.add(LSTM(32, input_shape=(train_x.shape[1], 1)))
	net.add(Dense(5, activation='softmax'))
	net.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
	net.fit(train_tensor, cat_train)
	print(net.evaluate(test_tensor, cat_test))

if __name__ == "__main__":
	xfile = "x_data.npy"
	yfile = "y_data.npy"
	train_x, train_y, test_x, test_y = load_data(xfile, yfile, 0.85)
	classifyLogistic(train_x, train_y, test_x, test_y)
	print();
	classifySVM(train_x, train_y, test_x, test_y)
	print()
	classifyNN(train_x, train_y, test_x, test_y)
	print()
	classifyCNN(train_x, train_y, test_x, test_y)
	print()
	classifyRNN(train_x, train_y, test_x, test_y)
