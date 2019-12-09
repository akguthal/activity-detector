""" IoT Project: Detection of Activity from Sensor Data """


# Libraries 
import numpy as np 
import pandas as pd
import scipy.io 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


""" Class for Activity Recognition from Accelerometer Data """
class ActivityDetect:
	# Initialisation Function
	def __init__(self):
		# Training Data
		self.training_data = None
		self.training_labels = None

		# Test Data
		self.test_data = None
		self.test_labels = None

		return

	# Module for Split
	def split(self, data_array, data_labels, train_size = None):
		# Generate random numbers
		indexes = np.arange(0,len(data_array))

		# Shuffle the Indexes
		np.random.shuffle(indexes)

		# Shuffled Featurized Data
		shuffled_data = data_array[indexes]

		# Shuffled Labels
		shuffled_labels = data_labels[indexes].reshape(-1,1)

		# Training Size Initialisation
		if train_size == None:
			train_size = int(0.8*len(data_array))


		# Training Data
		train_data = shuffled_data[:train_size]
		train_labels = shuffled_labels[:train_size]

		# Test Data
		test_data = shuffled_data[train_size:]
		test_labels = shuffled_labels[train_size:]


		# Return the Training and the Test Set
		return train_data, train_labels, test_data, test_labels


	# Define the Training Model
	def model(self):
		# Check for error
		if self.training_data is None or self.training_labels is None or self.test_data is None or self.test_labels is None:
			raise Exception("Initialisation Error")


		# Current Training Data
		curr_training_data = self.training_data
		curr_training_labels = self.training_labels

		# Current Test Data
		curr_test_data = self.test_data
		curr_test_labels = self.test_labels


		# Train a classifier using the training data 
		clf = SVC(gamma='auto')
		clf.fit(curr_training_data, curr_training_labels)

		#print(clf.score(curr_training_data, curr_training_labels))

		# Predicted Labels
		labels = clf.predict(curr_test_data)


		return labels


	# Main Function
	def main(self):
		# Data Matrix
		data_mat = scipy.io.loadmat('data/acc_data.mat')

		# Data Array
		data_array = data_mat["acc_data"]

		# The raw data label contains : Label, Label of subject performing the activity, Number of Trials
		data_label_vector = scipy.io.loadmat('data/acc_labels.mat')["acc_labels"]

		# Data Labels
		data_labels = data_label_vector[:,0]

		# Training Data and Test Data
		train_data, train_labels, test_datas, test_label = self.split(data_array, data_labels)


		#print(data_array.shape)
		#pca = PCA(n_components=2)
		#dim = pca.fit_transform(data_array)
		#X = dim[:,0]
		#Y = dim[:,1]
		#plt.scatter(X,Y, c= data_labels)
		#plt.xlabel('Feature 1')
		#plt.ylabel('Feature 2')
		#plt.show()


		# Initialise the Variables
		self.training_data = train_data
		self.training_labels = train_labels
		self.test_data = test_datas
		self.test_labels = test_label

		# Call the model
		labels = self.model()


		# Full Data
		#full_data = scipy.io.loadmat('data/full_data.mat')
		#print(full_data)

		
		return labels



# Initialise with the object
activity_object = ActivityDetect()


# Call the detection function
final_labels = activity_object.main()

print(final_labels)


