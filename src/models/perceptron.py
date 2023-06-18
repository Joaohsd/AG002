import numpy as np
import matplotlib.pyplot as plt

class Perceptron: 
    def __init__(self, alpha:float, n_epochs:int):
        self.alpha = alpha
        self.n_epochs = n_epochs
        
    def input_function(self, x):
        return self.weights.T.dot(x)
    
    def activation_function(self, value):
        return 2 if value >= 0 else 1
    
    def predict(self, x):
        input = self.input_function(x)
        return self.activation_function(input)
    
    def train(self, x:np.ndarray, y:np.ndarray):
        self.weights = np.random.random(x.shape[1] + 1).reshape(-1,1)
        
        bias = np.ones(x.shape[0]).reshape(-1,1)
        X = np.concatenate((bias, x), axis=1)

        isError = True
        epochs = 0

        self.hist_error_train = []

        while epochs < self.n_epochs and isError:
            isError = False
            epochs_error = 0
            for xi, yi in zip(X, y):
                predicted = self.predict(xi)
                error = yi - predicted
                epochs_error += abs(error)
                if predicted != yi:
                    self.weights = self.weights + self.alpha * error * xi.reshape(-1, 1)
                    isError = True
            self.hist_error_train.append(float(epochs_error)/X.shape[0])  
            epochs += 1

        self.plotLoss(epochs)

    def test(self, x):
        ''' 
            Dado uma lista de x, submete-os à rede
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
        plt.savefig('src/models/images/loss_epochs_perceptron.png', dpi=600)
        plt.close()