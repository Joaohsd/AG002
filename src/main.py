from database.bc_database import BreastCancerDB
from preprocessing.preprocessing import Preprocessing

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from models.models import *

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
    perceptron(X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy())
    adaline(X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy())
    decisionTree(X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy())
    randomForest(X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy())
    knn(X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy())
    
if __name__ == '__main__':
    main()