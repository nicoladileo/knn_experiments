# Experiments with KNearestNeighbor #

## Preface ##
In pattern recognition and machine learning, the k-Nearest Neighbors algorithm (or k-NN for short) is a non-parametric method used for classification and regression. In both cases, the input consists of the k closest training examples in the feature space
[KNN on Wikipedia](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)

## Experiment ##
I started this work to compare the accuracy of the classification made by the KNN algorithm, in presence and absence of data normalization. For accuracy of the algorithm, we mean the percentage of instances correctly classified.
The code is written in Python using the library **sklearn** and the four used datasets have been downloaded from the popular repository UCI. 
* Iris dataset [link] (https://archive.ics.uci.edu/ml/datasets/Iris)
* EEG Eye State [link](https://archive.ics.uci.edu/ml/datasets/EEG+Eye+State)
* Seeds dataset [link](https://archive.ics.uci.edu/ml/datasets/seeds)
* Magic gamma telescope [link](https://archive.ics.uci.edu/ml/datasets/MAGIC+Gamma+Telescope)

In order to run code type **python knn_main.py** on terminal: the KNN is computed for K equal to 1,3,5,10.

## Results ##

* Accuracy for 1NN: 0.9533 - Accuracy for 1NN with normalization step: 0.9467
* Accuracy for 3NN: 0.9600 - Accuracy for 3NN with normalization step: 0.9400
* Accuracy for 5NN: 0.9600 - Accuracy for 5NN with normalization step: 0.9667
* Accuracy for 10NN: 0.9600 - Accuracy for 10NN with normalization step: 0.9533
==================================================
