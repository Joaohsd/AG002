import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from models.neural_networks.perceptron import Perceptron
from models.neural_networks.adaline import Adaline
from models.neural_networks.multilayer_perceptron import MLP

from sklearn.metrics import accuracy_score, confusion_matrix

def perceptron(X_train, X_test, y_train, y_test):
    # Perceptron
    perceptron = Perceptron(alpha=0.001, n_epochs=15000)
    perceptron.train(X_train, y_train)
    predictedPerceptron = perceptron.test(X_test)
    print(f'Accuracy score for Perceptron: {accuracy_score(y_test, predictedPerceptron)}')
    # Plot the confusion matrix.
    mat = confusion_matrix(y_test, predictedPerceptron)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=['1','2'], yticklabels=['1','2'], cmap="Blues")
    plt.xlabel('True label', fontsize=14)
    plt.ylabel('Predicted label', fontsize=14)
    plt.savefig('src/models/images/confusion_matrix_perceptron.png', dpi=600)
    plt.close()
    return perceptron

def adaline(X_train, X_test, y_train, y_test):
    # Adaline
    adaline = Adaline(alpha=0.001, n_epochs=15000, max_error=1e-12)
    adaline.train(X_train, y_train)
    predictedAdaline = adaline.test(X_test)
    print(f'Accuracy score for Adaline: {accuracy_score(y_test, predictedAdaline)}')
    # Plot the confusion matrix.
    mat = confusion_matrix(y_test, predictedAdaline)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=['1','2'], yticklabels=['1','2'], cmap="Blues")
    plt.xlabel('True label', fontsize=14)
    plt.ylabel('Predicted label', fontsize=14)
    plt.savefig('src/models/images/confusion_matrix_adaline.png', dpi=600)
    plt.close()
    return adaline

def multilayerPerceptron(X_train, X_test, y_train, y_test):
    # Adaline
    mlp = MLP(alpha=0.01, max_error=1e-7, neurons=[16, 8, 1])
    mlp.train(X_train, y_train)
    preditedMLP = mlp.test(X_test)
    print(f'Accuracy score for MLP: {accuracy_score(y_test, preditedMLP)}')
    # Plot the confusion matrix.
    mat = confusion_matrix(y_test, preditedMLP)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=['1','2'], yticklabels=['1','2'], cmap="Blues")
    plt.xlabel('True label', fontsize=14)
    plt.ylabel('Predicted label', fontsize=14)
    plt.savefig('src/models/images/confusion_matrix_MLP.png', dpi=600)
    plt.close()
    return mlp

def decisionTree(X_train, X_test, y_train, y_test):
    # Decision Tree
    decisionTree = DecisionTreeClassifier()
    decisionTree.fit(X_train, y_train)
    predictedDecisionTree = decisionTree.predict(X_test)
    print(f'Accuracy score for Decision Tree: {accuracy_score(y_test, predictedDecisionTree)}')
    # Plot the confusion matrix.
    mat = confusion_matrix(y_test, predictedDecisionTree)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=['1','2'], yticklabels=['1','2'], cmap="Blues")
    plt.xlabel('True label', fontsize=14)
    plt.ylabel('Predicted label', fontsize=14)
    plt.savefig('src/models/images/confusion_matrix_decisionTree.png', dpi=600)
    plt.close()
    return decisionTree

def randomForest(X_train, X_test, y_train, y_test):
    # Random Forest
    randomForest = RandomForestClassifier()
    randomForest.fit(X_train, y_train)
    predictedRandomForest = randomForest.predict(X_test)
    print(f'Accuracy score for Random Forest: {accuracy_score(y_test, predictedRandomForest)}')
    # Plot the confusion matrix.
    mat = confusion_matrix(y_test, predictedRandomForest)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=['1','2'], yticklabels=['1','2'], cmap="Blues")
    plt.xlabel('True label', fontsize=14)
    plt.ylabel('Predicted label', fontsize=14)
    plt.savefig('src/models/images/confusion_matrix_randomForest.png', dpi=600)
    plt.close()
    return randomForest

def knn(X_train, X_test, y_train, y_test):
    # K-Nearest Neighboor
    acc_train = []
    acc_test = []
    score_max = 0
    k_max = 0
    Kmax = 99
    for k in range(1, Kmax+1, 2):
        # Create an instance of Neighbours Classifier and fit the data.
        clf = KNeighborsClassifier(k, weights='distance')
        # Train the classifier.
        clf.fit(X_train, y_train)
        # Predict.
        predictedKnn = clf.predict(X_test)
        # Calculate score.
        score_test = clf.score(X_test, y_test)
        acc_test.append(score_test)
        score_train = clf.score(X_train, y_train)
        acc_train.append(score_train)    
        if(score_test > score_max):
            score_max = score_test
            k_max = k
    # Train with best k
    # Create an instance of Neighbours Classifier and fit the data.
    knn = KNeighborsClassifier(k_max, weights='distance')
    # Train the classifier.
    knn.fit(X_train, y_train)
    # Predict.
    predictedKnn = knn.predict(X_test)
    print(f'Accuracy score for KNN using {k_max} neighboors: {accuracy_score(y_test, predictedKnn)}')
    # Plot the confusion matrix.
    mat = confusion_matrix(y_test, predictedKnn)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=['1','2'], yticklabels=['1','2'], cmap="Blues")
    plt.xlabel('True label', fontsize=14)
    plt.ylabel('Predicted label', fontsize=14)
    plt.savefig('src/models/images/confusion_matrix_knn.png', dpi=600)
    plt.close()
    return knn