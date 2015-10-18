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

In order to run code type **python knn_main.py** on terminal.

## Results ##
==================================================
Run on ucidataset/iris.data dataset
Total samples:150
Accuracy for 1NN: 0.9533 - Accuracy for 1NN with normalization step: 0.9467
Accuracy for 3NN: 0.9600 - Accuracy for 3NN with normalization step: 0.9400
Accuracy for 5NN: 0.9600 - Accuracy for 5NN with normalization step: 0.9667
Accuracy for 10NN: 0.9600 - Accuracy for 10NN with normalization step: 0.9533
==================================================
Run on ucidataset/eeg_eye_state.data dataset
Total samples:14980
Accuracy for 1NN: 0.9761 - Accuracy for 1NN with normalization step: 0.8700
Accuracy for 3NN: 0.9718 - Accuracy for 3NN with normalization step: 0.8727
Accuracy for 5NN: 0.9666 - Accuracy for 5NN with normalization step: 0.8696
Accuracy for 10NN: 0.9521 - Accuracy for 10NN with normalization step: 0.8573
==================================================
Run on ucidataset/seeds_dataset.data dataset
Total samples:210
Accuracy for 1NN: 0.9190 - Accuracy for 1NN with normalization step: 0.9381
Accuracy for 3NN: 0.8762 - Accuracy for 3NN with normalization step: 0.9143
Accuracy for 5NN: 0.8762 - Accuracy for 5NN with normalization step: 0.9238
Accuracy for 10NN: 0.9048 - Accuracy for 10NN with normalization step: 0.9190
==================================================
Run on ucidataset/magic.data dataset
Total samples:19020
Accuracy for 1NN: 0.7800 - Accuracy for 1NN with normalization step: 0.8162
Accuracy for 3NN: 0.7987 - Accuracy for 3NN with normalization step: 0.8351
Accuracy for 5NN: 0.8062 - Accuracy for 5NN with normalization step: 0.8373
Accuracy for 10NN: 0.8065 - Accuracy for 10NN with normalization step: 0.8354
==================================================
