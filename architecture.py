
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import random
from math import pi
from sklearn.metrics import confusion_matrix
from scipy.stats import linregress
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Architecture:

    def __init__(self, X_train, X_test, y_train, y_test):
        print('Architecture Classifier')
        # train data
        self.X_train = np.array(X_train)
        # self.X_train = X_train
        self.y_train = np.array(y_train)
        # test data
        self.X_test = np.array(X_test)
        # self.X_test = X_test
        self.y_test = np.array(y_test)
        # exit()

    def sigmoid(self, x=0.0):
        return 1 / 1 + np.exp(-x)

    def soft_max(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def metrics(self, y_true=[], y_pred=[]):
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

    def prediction(self, X=[], Y=[], weight0=None, weight1=None):
        predictions = []
        for i in range(len(X)):
            o = []
            for j in range(len(X[i])):
                a = abs(weight0[j]-X[i][j])+abs(weight1[j]+X[i][j])
                o.append(a)
            print(sum(o), Y[i])
            predictions.append(1)
        return predictions

    def projection3d(self, data=[], y=[], it=-1):
        fig = plt.figure()  # figsize=(4, 4)
        ax = fig.add_subplot(111, projection='3d')
        ax.grid(False)
        xs = self.column(data, 0)
        ys = self.column(data, 1)
        zs = self.column(data, 2)
        for i in range(len(zs)):
            if y[i] == 1:
                zs[i] = zs[i] + 0
        ax.scatter(xs=xs, ys=ys, zs=zs, c=y)
        plt.savefig('./outputs/3d/f'+str(it)+'.png')

    def learning(self, X=[], Y=[]):
        indexes = {}
        for i in range(len(X)):
            x = X[i]
            for j in range(len(x)):
                if not j in indexes:
                    indexes[j] = []
                indexes[j].append((x[j], Y[i]))
        weight0 = []
        weight1 = []
        for i in indexes:
            c = {0: 0, 1: 0}
            w = {0: [], 1: []}
            for v, j in indexes[i]:
                c[j] += 1
                w[j].append(v)
            weight0.append(np.mean(w[0]))
            weight1.append(np.mean(w[1]))
        weight0 = self.soft_max(np.array(weight0))
        weight1 = np.array(weight1)
        return weight0, weight1

    def fit(self):
        predictions = []
        weight0, weight1 = self.learning(
            X=self.X_train, Y=self.y_train)
        predictions = self.prediction(
            X=self.X_test, Y=self.y_test, weight0=weight0, weight1=weight1)
        self.metrics(y_true=self.y_test, y_pred=predictions)
