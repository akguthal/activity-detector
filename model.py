""" IoT Project: Detection of Activity from Sensor Data """


# Libraries 
import numpy as np 
import pandas as pd
import scipy.io 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle
from collections import Counter


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
			train_size = int(0.9*len(data_array))


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


		# Store the test data
		np.save('test_data.npy', curr_test_data)
		np.save('test_labels.npy', curr_test_labels)


		# Train a classifier using the training data 
		clf = SVC(gamma='auto')
		clf.fit(curr_training_data, curr_training_labels)

		#print(clf.score(curr_training_data, curr_training_labels))

		# Store the model checkpoint
		filename = 'finalized_model.sav'
		pickle.dump(clf, open(filename, 'wb'))

		# Predicted Labels
		labels = []
		#labels = clf.predict(curr_test_data)


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


	# Function to test the model with data
	def prediction_function(self, test_data, test_labels):
		# Current filename
		filename = 'finalized_model.sav'

		# Loaded model
		loaded_model = pickle.load(open(filename, 'rb'))

		# Write the prediction function
		predictions = loaded_model.predict(test_data)

		# Final Labels 
		final_labels = predictions - 1

		# Data Labels
		label_names = scipy.io.loadmat('data/acc_names.mat')['acc_names']

		# Complete label names
		names = np.array([item[0] for item in label_names[0]])

		# Final Labels
		label_to_return = list(names[final_labels])

		# Distributions for the labels
		distributions = Counter(label_to_return)

		count_of_measurements = len(label_to_return)


		return distributions, count_of_measurements




# Main Function
def function_main(train = True):
	# Initialise with the object
	activity_object = ActivityDetect()

	# When train is true
	if train == True:
		# Call the detection function - For training
		final_labels = activity_object.main()

		return 

	# Condition when the train is False
	else:
		# Test Training Data
		test_training_data = np.load('data/test_data.npy')

		# Test Labelled Data
		test_labelled_data = np.load('data/test_labels.npy')

		# Call the prediction function - with the training data and test data
		distributions, count_of_measurements = activity_object.prediction_function(test_training_data, test_labelled_data)

		print("Total Measurements: ")
		print(count_of_measurements)



		print(distributions)
		return distributions




function_main(False)




