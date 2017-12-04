# label-segments.py
# Andrew Kramer

# organizes the variable-length data samples into data streams 
# and assigns labels
# splits into training and testing sets and saves as a .npy file

# labels:
# 0 - neutral
# 1 - fist
# 2 - down
# 3 - up
# 4 - left
# 5 - right

import numpy as np

def load_filenames(name_file_list):
	name_files = np.loadtxt(name_file_list, dtype=str)
	filename_list = [];
	for name_file in name_files:
		filenames = np.loadtxt(name_file, dtype=str)
		prefix = 'Segmented_Data/'
		if 'up' in name_file:
			prefix = prefix + 'up/'
		if 'down' in name_file:
			prefix = prefix + 'down/'
		if 'left' in name_file:
			prefix = prefix + 'left/'
		if 'right' in name_file:
			prefix = prefix + 'right/'
		if 'fist' in name_file:
			prefix = prefix + 'fist/'
		for name in filenames:
			filename_list.append(prefix + name) 
	return filename_list

def load_data(filename_list):
	examples = [];
	labels = [];
	sum_samples = 0
	for filename in filename_list:
		example = np.genfromtxt(filename, delimiter=',')
		sum_samples += np.size(example,0)
		examples.append(example)
		if 'fist' in filename:
			labels.append(1)
		if 'down' in filename:
			labels.append(2)
		elif 'up' in filename:
			labels.append(3)
		elif 'left' in filename:
			labels.append(4)
		elif 'right' in filename:
			labels.append(5)
	return examples, labels, sum_samples

def shuffle_data(examples, labels):
	num_examples = len(examples)
	indices = np.array(range(0, num_examples))
	np.random.shuffle(indices)
	shuf_examples = []
	shuf_labels = []
	for i in indices:
		shuf_examples.append(examples[i])
		shuf_labels.append(labels[i])
	return shuf_examples, shuf_labels

def create_stream(examples, labels, num_rows):
	example_stream = np.zeros((num_rows,16))
	label_stream = np.zeros(num_rows)
	str_index = 0 # current index in the data stream
	for i in range(0,len(examples)):
		ex_len = np.size(examples[i],0) # overall length of example
		n_len = int(ex_len/8) # length of neutral gesture before 
								# and after labeled gesture occurs
		# add labels to label stream
		label_stream[str_index + n_len:str_index + ex_len - n_len] = labels[i];
		# add example to stream
		example_stream[str_index:str_index + ex_len] = examples[i]

		str_index += ex_len
	return example_stream, label_stream

if __name__ == "__main__":
	print('loading filenames')
	filename_list = load_filenames('seg_names.txt')
	print('loading data from files')
	examples, labels, n_samples = load_data(filename_list)
	print('shuffling data')
	examples, labels = shuffle_data(examples, labels)
	print('creating data streams')
	example_stream, label_stream = create_stream(examples, labels, n_samples)
	np.save('x_stream.npy', example_stream)
	np.save('y_stream.npy', label_stream)
