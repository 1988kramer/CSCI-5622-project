import argparse
import numpy as np

# labels:
# 0 - fist
# 1 - down
# 2 - up
# 3 - left
# 4 - right



def save_files(filenames):
	x_data = np.zeros(shape=(2111,10,16))
	y_data = np.zeros(2111)
	i = 0
	for file in filenames:
		file = 'data/' + file;
		raw_data = np.genfromtxt(file, delimiter=',')
		truncate = raw_data.shape[0] - (raw_data.shape[0] % 10)
		raw_data = raw_data[0:truncate,0:16]
		shaped_data = np.reshape(raw_data, (-1, 10, 16))
		n_file = shaped_data.shape[0]
		x_data[i:i+n_file, :, :] = shaped_data
		if 'down' in file:
			y_data[i:i+n_file] = 1
		elif 'up' in file:
			y_data[i:i+n_file] = 2
		elif 'left' in file:
			y_data[i:i+n_file] = 3
		elif 'right' in file:
			y_data[i:i+n_file] = 4
		i += n_file
	np.save('x_data.npy', x_data)
	np.save('y_data.npy', y_data)

def load_names(file):
	files = np.loadtxt(file, dtype=str)
	return files

if __name__ == "__main__":
	filenames = load_names('datafiles.txt')
	save_files(filenames)
