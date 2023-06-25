import numpy as np
import matplotlib.pyplot as plt

class Adaline: 
    def __init__(self, alpha:float, n_epochs:int, max_error:float):
        self.alpha = alpha
        self.n_epochs = n_epochs
        self.error = max_error
        
    def input_function(self, x):
        return self.weights.T.dot(x)
    
    def activation_function(self, value):
        return 2 if value >= 1.5 else 1
    
    def predict(self, x):
        input = self.input_function(x)
        return self.activation_function(input)
    
    def train(self, x:np.ndarray, y:np.ndarray):
        self.weights = np.random.random(x.shape[1] + 1).reshape(-1,1)
        
        bias = np.ones(x.shape[0]).reshape(-1,1)
        X = np.concatenate((bias, x), axis=1)

        epochs = 0        
        mse = 0
        last_mse = np.inf
        self.hist_error_train = []

        while epochs < self.n_epochs:
            mse = 0
            
            for xi, yi in zip(X, y):
                predicted = self.input_function(xi)
                error = yi - predicted
                mse += (error ** 2)
                self.weights = self.weights + self.alpha * error * xi.reshape(-1, 1)
            
            mse = (float(mse)/X.shape[0])
            #print(f"EPOCH: {epochs}\t- MSE: {mse}\t- MSE_ant - MSE: {abs(last_mse - mse)}")
            
            if abs(last_mse - mse) <= self.error:
                break
            
            last_mse = mse
            self.hist_error_train.append(mse)
            epochs += 1
        self.plotLoss(epochs)

    def test(self, x):
        ''' 
            List of x and submit them to the neural network
        '''
        bias = np.ones(x.shape[0]).reshape(-1,1)
        X = np.concatenate((bias, x), axis=1)
        results = []
        for xi in X:
            predict = self.predict(xi)
            results.append(predict)
            
        return results
    
    def plotLoss(self, epochs):
        plt.figure(figsize=(8,5))
        plt.plot(np.arange(start=1, stop=epochs+1, step=1), self.hist_error_train, color='red')
        plt.title('Error')
        plt.xlabel('Epochs')
        plt.ylabel('MSE')
        plt.savefig('src/models/images/loss_epochs_adaline.png', dpi=600)
        plt.close()