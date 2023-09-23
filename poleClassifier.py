
import numpy as np
from math import cos, pi, sqrt, sin, exp, tanh,  log2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix

# computing of weights and error that have least of intrus
#
class PoleClassifier:

    def __init__(self, X_train, X_test, y_train, y_test):
        print('Pole Classifier')
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    
    def draw_cuve(self, x, y, title, label_x, label_y):
        plt.figure(figsize=(8, 6))
        plt.plot(x, y, label=label_y)
        plt.title(title)
        plt.xlabel(label_x)
        plt.ylabel(label_y)
        plt.legend()
        plt.grid(True)
        plt.savefig('./outputs/learning_error.png')
        print('End')

    def metrics(self, y_true=[] , y_pred=[]):
        a = round(accuracy_score(y_true, y_pred), 2)
        p = round(precision_score(y_true, y_pred), 2)
        r = round(recall_score(y_true, y_pred), 2)
        f = round(f1_score(y_true, y_pred), 2)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        print('Metrics : ')
        print('Accuracy : ', a)
        print('Precision : ', p)
        print('Recall : ', r)
        print('F1-score : ', f)
        print(f'Confusion matrix : tn({tn}), fp({fp}), fn({fn}), tp({tp})')
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def prediction(self, X=[], Y=[], weight=[], error=0.0):
        Y_pred = []
        fp = 0
        for i in range(len(X)):
            n = len(X[i])
            sinusoidal = []
            for j in range(n):
                x = X[i][j]
                # value = (x*np.cos(sqrt(2)*x*pi + 1) + np.sin(3*np.pi*np.tanh((j+1)*x))) / np.exp(n)
                # value = x*np.cos(sqrt(2)*x*pi + 1)/(1+np.exp(n))
                value = np.cos(2*np.pi + x**2) + np.tanh(2*x)/ (1+np.exp(abs(x)+1))
                sinusoidal.append(abs(value))
            sinusoidal = np.array(sinusoidal)
            prediction = self.sigmoid(np.dot(sinusoidal, weight))
            _error = abs(1-prediction)
            if  _error < sqrt(error) :
                Y_pred.append(1)
            else:
                Y_pred.append(0)
        return Y_pred
    
    def learn_positive(self, Xg=[], Xb=[], Y=[], learning_rate=0.01, epochs=60):
        error = 0.0
        std = 0.0
        learning_rate=0.0001
        epochs=100
        errors = []
        intrusions = []
        _weights = []
        n = len(Xg[0])
        print('Feature size : ', n)
        weights = np.array([ random.uniform(0, 1) for _ in range(n)])
        for _ in range(epochs):
            _errors = []
            for i in range(len(Xg)):
                n = len(Xg[i])
                sinusoidal = []
                for j in range(n):
                    x = Xg[i][j]
                    value = np.cos(2*np.pi + x**2) + np.tanh(2*x)/(1+np.exp(abs(x)+1))  # / (1+np.exp(sqrt(n)))
                    # print(value)
                    # sqrt(2)*x*np.cos(3*np.pi + x**2) 
                    # - x*np.sin(sqrt(3)*np.pi*np.tanh(x))) / np.exp(n)
                    sinusoidal.append(abs(value))
                sinusoidal = np.array(sinusoidal)
                prediction = self.sigmoid(np.dot(sinusoidal, weights))
                _error = abs(Y[i] - prediction)
                weights += learning_rate * _error * Xg[i]
                _errors.append(_error)
            # print('Mean : ', np.mean(_errors))
            # exit()
            error = np.mean(_errors)
            errors.append(error)
            _weights.append(weights)
            # intrusions.append(self.intrusion(Xb=Xb, weight=weights, error=error))
        self.draw_cuve([i for i in range(len(errors))], errors, 'Learning errors', 'epochs', 'errors')
        # print('Count of bad : ', len(Xb))
        # print(intrusions)
        # min_error = min(errors)
        # index = errors.index(min_error)
        error = errors[-1] # errors[index]
        weights = _weights[-1] # _weights[index]
        print('Weights : ', weights)
        print('Error :', error)
        return weights, error

    def intrusion(self, Xb=[], weight=[], error=0.0):
        output = len(Xb)
        for i in range(len(Xb)):
            sinusoidal = np.array([ sqrt(2)*x*cos(2*x*pi + 1) + exp(x**2)*pi + 1 for x in Xb[i] ])
            prediction = self.sigmoid(np.dot(sinusoidal, weight))
            _error = abs(1 - prediction)
            if _error < error :
                output = output - 1
        return output
    
    def fit(self, learning_rate=0.001, epochs=60):
        predictions = []
        positives_x = []
        positives_y = []

        negatives_x = []
        negatives_y = []

        for i in range(len(self.y_train)):
            if self.y_train[i] == 1 :
                positives_x.append(self.X_train[i])
                positives_y.append(self.y_train[i])
            else :
                negatives_x.append(self.X_train[i])
                negatives_y.append(self.y_train[i])
        weights, error = self.learn_positive(Xg=positives_x, Xb=negatives_x, Y=positives_y, learning_rate=learning_rate, epochs=epochs)

        # predictions = self.prediction(X=positives_x, Y=positives_y, weight=weights, error=error)
        # self.metrics(y_true=positives_y, y_pred=predictions)
        
        print('Learning : ')
        predictions = self.prediction(X=self.X_train, Y=self.y_train, weight=weights, error=error)
        self.metrics(y_true=self.y_train, y_pred=predictions)
        print('Testing : ')
        predictions = self.prediction(X=self.X_test, Y=self.y_test, weight=weights, error=error)
        self.metrics(y_true=self.y_test, y_pred=predictions)

