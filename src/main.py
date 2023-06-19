from database.bc_database import BreastCancerDB
from preprocessing.preprocessing import Preprocessing

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from models.models import *
from cli.menu import MenuCLI

def main():
    print('Succesfully started!')
    
    host = '127.0.0.1'
    port = 3306
    user = 'root'
    password = 'root'
    database = 'ag002'

    np.random.seed(1000)

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

    # Preprocessing using Oversampling in order to balance Dataset
    X_resampled, y_resampled = Preprocessing.resampleDataset(X, y)

    # Split dataset into train and test parts in order to train and verify model performance
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.20, shuffle=True)

    # Train models and verify performance by accuracy and confusion matrix
    _perceptron = perceptron(X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy())
    _adaline = adaline(X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy())
    #multilayerPerceptron(X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy())
    _decision_tree = decisionTree(X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy())
    _random_forest = randomForest(X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy())
    _knn = knn(X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy())
    
    # Models
    models = []
    models.append(_perceptron)
    models.append(_adaline)
    models.append(_decision_tree)
    models.append(_random_forest)
    models.append(_knn)

    # Menu
    menu = MenuCLI(models)
    menu.run()

if __name__ == '__main__':
    main()