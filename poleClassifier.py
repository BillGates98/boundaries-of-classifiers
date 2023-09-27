
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import random
from math import pi
from sklearn.metrics import confusion_matrix
from bearer import Wave
import copy
# computing of weights and error that have least of intrus
#
class PoleClassifier:

    def __init__(self, X_train, X_test, y_train, y_test):
        print('Pole Classifier')
        self.X_train = [ self.normalize_features(X) for X in X_train]
        self.X_test = [ self.normalize_features(X) for X in X_test]
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
        plt.savefig('./outputs/learning_error_' + str(label_y) + '.png')
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
    
    def normalize_features(self, features):
        if not isinstance(features, (list, np.ndarray)):
            raise ValueError("Input 'features' must be a list or numpy array.")
        features = np.array(features)
        mean = np.mean(features)
        std_dev = np.std(features)
        if std_dev == 0:
            return features
        normalized_features = (features - mean) / std_dev
        return normalized_features

    def prediction(self, X=[], Y=[], weights=[]):
        Y_preds = []
        print('Decision Status : 0 -' ,  1 in Y  , ' and 1-',  0 in Y )
        preds = np.dot(X, weights)
        for i in range(len(preds)):
            tmp = round(preds[i])
            if tmp == Y[i] :
                Y_preds.append(tmp)
            else:
                # print(tmp, ' ---------->>>>>>> ', Y[i])
                Y_preds.append(1 - tmp)
        return Y_preds
    
    def mean_squared_error(self, y_true=[], y_pred=[]):
        return np.mean((y_true - y_pred)**2)

    def learn_positive(self, X=[], Y=[]):
        print(' Learning Status : 1-' ,  1 in Y  , ' and 0-',  0 in Y )
        learning_rate = 0.00001
        errors = []
        _N = len(X[0])
        weights = np.random.rand(_N)
        errors = []
        i = 100000
        while i > 0:
            y_pred = np.dot(X, weights)
            # print(y_pred)
            # exit()
            error = self.mean_squared_error(Y, y_pred)
            if error < 0.000000001:
                break
            gradient = +3 * np.dot(X.T, (Y - y_pred))
            weights += learning_rate * gradient
            errors.append(error)
            i = i - 1 
        predictions = self.prediction(X=self.X_train, Y=self.y_train, weights=weights)
        # predictions = self.prediction(X=X, Y=Y, weights=weights)
        # print(predictions)
        # print(Y)
        # exit()
        # self.metrics(y_true=Y, y_pred=predictions)
        self.metrics(y_true=self.y_train, y_pred=predictions)
        self.draw_cuve([i for i in range(len(errors))], [f for f in errors], 'Learning errors', 'epochs', 'g_errors')
        return weights
    
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

        # print('All \n')
        # weights1 = self.learn_positive(X=np.array(self.X_train), Y=np.array(self.y_train))
        # print('Positives \n')
        # weights2 = self.learn_positive(X=np.array(positives_x), Y=np.array(positives_y))
        print('Positives Learning on Negatives data \n')
        weights3 = self.learn_positive(X=np.array(negatives_x), Y=np.array(negatives_y))

        # predictions = self.prediction(X=positives_x, Y=positives_y, weights=weights)
        # self.metrics(y_true=positives_y, y_pred=predictions)
        
        # print('Learning : ')
        # predictions = self.prediction(X=self.X_train, Y=self.y_train, weights=weights)
        # self.metrics(y_true=self.y_train, y_pred=predictions)

        # predictions = self.prediction(X=self.X_train, Y=self.y_train, weights=weights)
        # self.metrics(y_true=self.y_train, y_pred=predictions)

        print('Testing : ')
        predictions = self.prediction(X=self.X_test, Y=self.y_test, weights=weights3)
        self.metrics(y_true=self.y_test, y_pred=predictions)

