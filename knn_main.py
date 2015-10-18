""" 
Copyright (C) 2015  Nicola Dileo
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>


Module: knn_main.py
--------
"""

from sklearn.cross_validation import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler 
from knn_io import load_dataset
import numpy as np

def start_experiment(dataset):
	data = load_dataset(dataset)	
	print("Run on %s dataset"%(data.DESCR))	
	print("Total samples:%d"%len(data.data))
	print("="*50)
	acc1,acc_norm1 = execute_KNN(data,1,5)  #Execute 1NN
	acc3,acc_norm3 = execute_KNN(data,3,5)  #Execute 3NN
	acc5,acc_norm5 = execute_KNN(data,5,5)  #Execute 5NN	
	acc10,acc_norm10 = execute_KNN(data,10,5) #Execute 10NN

	print("> Accuracy for 1NN: %.4f - Accuracy for 1NN with normalization step: %.4f"%(acc1,acc_norm1))
	print("> Accuracy for 3NN: %.4f - Accuracy for 3NN with normalization step: %.4f"%(acc3,acc_norm3))
	print("> Accuracy for 5NN: %.4f - Accuracy for 5NN with normalization step: %.4f"%(acc5,acc_norm5))
	print("> Accuracy for 10NN: %.4f - Accuracy for 10NN with normalization step: %.4f"%(acc10,acc_norm10))
	print("="*50 + "\n")


def execute_KNN(dataset, neighbors, folds):
	features = dataset.data
	labels = dataset.target
	means = []
	means_norm = []
	kf = KFold(len(features),n_folds = folds, shuffle = True)
	classifier = KNeighborsClassifier(n_neighbors = neighbors)  #Classic classifier
	classifier_norm = KNeighborsClassifier(n_neighbors = neighbors)  #Classifier with normalization step
	classifier_norm = Pipeline([('norm', StandardScaler()),('knn', classifier_norm)])

	for training,testing in kf:
		#fit model to training set
		classifier.fit(features[training],labels[training])
		classifier_norm.fit(features[training],labels[training])

		#predict data in test set
		prediction = classifier.predict(features[testing])
		prediction_norm = classifier_norm.predict(features[testing])		
		
		#compute accuracy as mean of correct classified instances
		accuracy = np.mean(prediction == labels[testing])
		accuracy_norm = np.mean(prediction_norm == labels[testing])
		means.append(accuracy)
		means_norm.append(accuracy_norm)
	
	return np.mean(means), np.mean(means_norm)


if __name__ == '__main__':
	start_experiment('ucidataset/iris.data')
	start_experiment('ucidataset/eeg_eye_state.data')
	start_experiment('ucidataset/seeds_dataset.data')
	start_experiment('ucidataset/magic.data')
	
	


