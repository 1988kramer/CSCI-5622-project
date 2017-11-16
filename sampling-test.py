# labels:
# 0 - fist
# 1 - down
# 2 - up
# 3 - left
# 4 - right

import numpy as np
from sklearn import linear_model, preprocessing
from sklearn.svm import SVC
from scipy.signal import resample
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

def load_data(xfile, yfile):
	x_data = np.load(xfile) # load ordered x values
	y_data = np.load(yfile) # load ordered y values
	return x_data, y_data

# returns the mean of the given data for each sensor reading
def mean_x(x_data, m):
	m = np.size(x_data, 0)
	n_samples = np.size(x_data, 1)
	x_data = np.sum(x_data, 1)
	x_data = x_data / n_samples
	return x_data

# randomly up or downsamples the given data and then 
# returns the mean of the resampled data
def mean_resample_x(x_data, m):
	m = np.size(x_data, 0)
	resampled_x = np.zeros((m,16))
	max_samples = 20
	min_samples = 5
	# randomly up or downsample to between 0.5 and 2s
	# then take mean
	for i in range(0, m):
		this_x = resample(x_data[i],
						  np.random.randint(min_samples,max_samples), 
						  axis=0)
		samples = np.size(this_x, 0)
		this_x = np.sum(this_x, 0) # change to mean of all samples for each value
		resampled_x[i] = this_x / samples

	return resampled_x

def resample_x(x_data, samples, m):
	x_data = resample(x_data, samples, axis=1) # resample to specified size
	x_data = np.reshape(x_data, (m,-1)) # reshape to 2-D vector
	return x_data


# shuffles data and splits into test and training sets
def shuffle_data(x_data, y_data, split):
	m = np.size(y_data)
	# shuffle x and y values
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

	logreg = linear_model.LogisticRegression(multi_class='multinomial', 
											 solver='saga', C=100, 
											 max_iter=5000)


	logreg.fit(train_x, train_y)
	pred = logreg.predict(test_x)
	print("Logistic regression results: ")
	print(classification_report(test_y, pred))

def classifySVM(train_x, train_y, test_x, test_y):
	poly = SVC(kernel='poly', degree=3, C=100, decision_function_shape='ovo')
	poly.fit(train_x, train_y)
	y_pred = poly.predict(test_x)
	print("Polynomial SVM results:")
	print(classification_report(test_y, y_pred))
	print()


if __name__ == "__main__":
	xfile = "x_data.npy"
	yfile = "y_data.npy"
	x_data, y_data = load_data(xfile, yfile)
	m = np.size(x_data, 0)
	#x_data = resample_x(x_data, 5, m)
	#x_data = mean_x(x_data, m)
	x_data = mean_resample_x(x_data, m)
	#x_data = np.reshape(x_data, (m,-1)) # reshape to 2-D vector
	x_data = preprocessing.scale(x_data) # normalize data
	train_x, train_y, test_x, test_y = shuffle_data(x_data, y_data, 0.85)
	classifyLogistic(train_x, train_y, test_x, test_y)
	print()
	classifySVM(train_x, train_y, test_x, test_y)