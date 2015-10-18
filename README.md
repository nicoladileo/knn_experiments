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
Iris dataset: 150 samples
* Accuracy for 1NN: 0.9533 - Accuracy for 1NN with normalization step: 0.9467
* Accuracy for 3NN: 0.9600 - Accuracy for 3NN with normalization step: 0.9400
* Accuracy for 5NN: 0.9600 - Accuracy for 5NN with normalization step: 0.9667
* Accuracy for 10NN: 0.9600 - Accuracy for 10NN with normalization step: 0.9533

EEG Eye State: 14980 samples
* Accuracy for 1NN: 0.9761 - Accuracy for 1NN with normalization step: 0.8700
* Accuracy for 3NN: 0.9718 - Accuracy for 3NN with normalization step: 0.8727
* Accuracy for 5NN: 0.9666 - Accuracy for 5NN with normalization step: 0.8696
* Accuracy for 10NN: 0.9521 - Accuracy for 10NN with normalization step: 0.8573

Seeds dataset: 210 samples
* Accuracy for 1NN: 0.9190 - Accuracy for 1NN with normalization step: 0.9381
* Accuracy for 3NN: 0.8762 - Accuracy for 3NN with normalization step: 0.9143
* Accuracy for 5NN: 0.8762 - Accuracy for 5NN with normalization step: 0.9238
* Accuracy for 10NN: 0.9048 - Accuracy for 10NN with normalization step: 0.9190

Magic telescope dataset: 19020 samples
* Accuracy for 1NN: 0.7800 - Accuracy for 1NN with normalization step: 0.8162
* Accuracy for 3NN: 0.7987 - Accuracy for 3NN with normalization step: 0.8351
* Accuracy for 5NN: 0.8062 - Accuracy for 5NN with normalization step: 0.8373
* Accuracy for 10NN: 0.8065 - Accuracy for 10NN with normalization step: 0.8354

As you can see the algorithm with data normalization better performs on dataset with different scale for attribute (Seeds and Magic dataset); on the other hand, if the data normalization is performed on dataset with same scale for attribute (Iris and EEG dataset), the performance are worst. 

