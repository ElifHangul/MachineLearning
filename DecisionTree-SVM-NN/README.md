# Classification

Binary classification dataset is used for decision trees and support vector machines(SVM):
https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#breast-cancer 

Binary classification dataset used for neural networks:
https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#diabetes 



# Decision Trees

Decision tree learning algorithm is run on the dataset.
Total classification error is reported. 
Experiment is repeated with different partitions and the resulting trees are plotted.


<p align="center">
<img src="https://github.com/ElifHangul/MachineLearning/blob/master/DecisionTree-SVM-NN/images/decision_tree.png" width=600 height=400>
</p>


# Support Vector Machines(SVM)

SVM is run to train a classifier, using radial basis as kernel function.
Cross validation is applied to evaluate different cobinations of values of the model hyper-parameters (C and gamma).
The combination of C and gamma that minimizes the cross validation error is selected and SVM is trained on whole set. Total classification error is reported.


<p align="center">
<img src="https://github.com/ElifHangul/MachineLearning/blob/master/DecisionTree-SVM-NN/images/svm_graph.png" height=400>
</p>

# Neural Networks(NN)

Multi-Layer perceptron using the cross-entropy loss with l-2 regularization (weight decay penalty) is trained.
Curves of the training and validation error as a function of the penalty strength is plotted.
Logaritmic range for hyper-parameter alpha is used. Experimented with different sizes of the training/validation sets and different model parameters(network layers).
Behavior of the curves are explained.
<br>
<p align="center">
<img src="https://github.com/ElifHangul/MachineLearning/blob/master/DecisionTree-SVM-NN/images/mlp_graph.png" height=400>
</p>
