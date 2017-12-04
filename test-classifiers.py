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

if __name__ == "__main__":
	xfile = "x_data.npy"
	yfile = "y_data.npy"
	train_x, train_y, test_x, test_y = load_data(xfile, yfile, 0.85)
	classifyLogistic(train_x, train_y, test_x, test_y)
	print();
	classifySVM(train_x, train_y, test_x, test_y)