from database.bc_database import BreastCancerDB
from preprocessing.preprocessing import Preprocessing

from models.perceptron import Perceptron
from models.adaline import Adaline

import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

def perceptron(X_train, X_test, y_train, y_test):
    # Perceptron
    perceptron = Perceptron(alpha=0.05, n_epochs=10000)
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

def adaline(X_train, X_test, y_train, y_test):
    # Adaline
    adaline = Adaline(alpha=0.01, n_epochs=10000, max_error=1e-12)
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
    clf = KNeighborsClassifier(k, weights='distance')
    # Train the classifier.
    clf.fit(X_train, y_train)
    # Predict.
    predictedKnn = clf.predict(X_test)
    print(f'Accuracy score for KNN using {k_max} neighboors: {accuracy_score(y_test, predictedKnn)}')
    # Plot the confusion matrix.
    mat = confusion_matrix(y_test, predictedKnn)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=['1','2'], yticklabels=['1','2'], cmap="Blues")
    plt.xlabel('True label', fontsize=14)
    plt.ylabel('Predicted label', fontsize=14)
    plt.savefig('src/models/images/confusion_matrix_knn.png', dpi=600)
    plt.close()

def main():
    print('Succesfully started!')
    
    host = '127.0.0.1'
    port = 3306
    user = 'root'
    password = 'root'
    database = 'ag002'

    # Accessing Database and getting information to train Machine Learning models
    bcDatabase = BreastCancerDB(host=host, port=port, user=user, password=password, database=database)
    values = bcDatabase.selectAll()
    columnNames = bcDatabase.selectColumnNames()

    # Creating Dataframe
    df = pd.DataFrame(values)
    df.columns = columnNames

    # Analyzing data
    Preprocessing.covariationMatrix(df)
    Preprocessing.correlationMatrix(df)
    Preprocessing.verifyDatasetBalance(df.iloc[:,-1])

    # Gettting X and y values to train models
    X = df.iloc[:,1:-1].to_numpy()
    y = df.iloc[:,-1].to_numpy()

    # Split dataset into train and test parts in order to train and verify model performance
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True)

    # Train models and verify performance by accuracy and confusion matrix
    perceptron(X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy())
    adaline(X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy())
    decisionTree(X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy())
    randomForest(X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy())
    knn(X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy())
    
if __name__ == '__main__':
    main()