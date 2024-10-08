import numpy as np
import pandas as pd


class Test:

    def __init__(self):
        print('Test')
        self.N = 9
        self.X = [(i, i % 2) for i in range(0, 10, 1)]
        self.matrix = [[0 for i in range(self.N+1)]
                       for j in range(self.N+1)]

    def remove_duplicates(self, lst=[]):
        return [t for t in (set(tuple(i) for i in lst))]

    def decomposition(self, pairs=[], n=0):
        output = []
        for a, b in pairs:

        return output

    def best_pair(self, v=0):
        output = []
        return output

    def prime_factors(self, n=0):
        output = []
        tmp = []
        for i in range(0, self.N+1):
            tmp.append(i)
        for i in range(len(tmp)):
            for j in range(len(tmp)):
                if tmp[i] + tmp[j] == n:
                    output.append((tmp[i], tmp[j]))
        return output

    def show_matrix(self, data=[]):
        print(pd.DataFrame(np.array(data)))

    def train_matrix(self, data=[]):
        matrix = self.matrix
        pairs = []
        for x, pf, c in data:
            for i, j in pf:
                matrix[i][j] += c
                pairs.append((i, j))
        self.show_matrix(data=matrix)
        return matrix, pairs

    def complete_training(self, data=[], pairs=[]):
        matrix = data
        for i in range(1, len(data)):
            for j in range(1, len(data)):
                prime_factors = self.depth_prime_factors(n=i)
                print(i, j, ' #> ')

    # def prediction(self):

    def run(self):
        data = []
        print(' 1 : pair')
        print(' -1 : impair')
        for x, c in self.X:
            print((x, self.prime_factors(n=x), c))
            data.append((x, self.prime_factors(n=x), -1 if c == 1 else 1))
        matrix, pairs = self.train_matrix(data=data)
        print('\n Completion process : \n ')
        self.complete_training(data=matrix, pairs=pairs)


Test().run()
