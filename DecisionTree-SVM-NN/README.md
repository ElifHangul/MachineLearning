# Classification

https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#breast-cancer binary classification dataset is used for decision trees and support vector machines(SVM)


https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#diabetes binary classification dataset used for neural networks.



# Decision Trees

Decision tree learning algorithm is run on the dataset.
Total classification error is reported. 
Experiment is repeated with different partitions and the resulting trees are plotted.


<p align="center">
<img src="https://github.com/ElifHangul/MachineLearning/blob/master/DecisionTree-SVM-NN/images/d02-7.png" height=400>
</p>

<br>
# Support Vector Machines(SVM)

SVM is run to train a classifier, using radial basis as kernel function.
Cross validation is applied to evaluate different cobinations of values of the model hyper-parameters (C and gamma).
The combination of C and gamma that minimizes the cross validation error is selected and SVM is trained on whole set. Total classification error is reported.


<p align="center">
<img src="https://github.com/ElifHangul/MachineLearning/blob/master/DecisionTree-SVM-NN/images/svm.png" height=400>
</p>

<br>
# Neural Networks(NN)

Multi-Layer perceptron using the cross-entropy loss with l-2 regularization (weight decay penalty) is trained.
Curves of the training and validation error as a function of the penalty strength is plotted.
Logaritmic range for hyper-parameter alpha is used. Experimented with different sizes of the training/validation sets and different model parameters(network layers).
Behavior of the curves are explained.
<br>
<p align="center">
<img src="https://github.com/ElifHangul/MachineLearning/blob/master/DecisionTree-SVM-NN/images/mlp03-10.png" height=400>
</p>
