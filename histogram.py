from compute_files import ComputeFile
import time
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker


class Main:

    def __init__(self, input_path='', output_path='', suffix='', model=''):
        self.input_path = input_path
        self.output_path = output_path
        self.suffix = suffix
        self.model = model
        self.dimensions = [i for i in range(100, 300, 50)]
        self.classifiers = ["AdaBoostClassifier", "RandomForestClassifier",    "XGBClassifier",    "ExtraTreeClassifier",
                            "LogisticRegression", "SVC", "KNeighborsClassifier", "DecisionTreeClassifier",  "GaussianNB"]  # ,    "GradientBoostinglassifier"]
        self.class_by_dim = {}
        self.short_classifier = {
            'AdaBoostClassifier': 'Ada',
            'RandomForestClassifier': 'Ran',
            'XGBClassifier': 'XGB',
            'ExtraTreeClassifier': 'Ext',
            'LogisticRegression': 'LoR',
            'SVC': 'SVC',
            'KNeighborsClassifier': 'KNN',
            'DecisionTreeClassifier': 'DeT',
            'GaussianNB': 'GNB'
        }
        self.models = {
            'r2v': 'RDF2Vec',
            'RESCAL': 'RESCAL',
            'TransE': 'TransE',
            'DistMult': 'DistMult',
        }
        self.start_time = time.time()

    def plot_data(self, data=[], metric=''):
        classifiers = self.classifiers
        dimensions = ['dim-'+str(i) for i in range(100, 300, 50)]
        values = np.array(data)
        n = len(values)
        w = .15
        x = np.arange(0, len(classifiers))
        colors = ['#007acc', '#00b386', '#ff6b6b', '#ffc107', '#aa80ff']
        for i, value in enumerate(values):
            position = x + (w*(1-n)/2) + i*w
            plt.bar(position, value, width=w,
                    label=f'{dimensions[i]}', color=colors[i])

        plt.xticks(x, [str(i+1) + '.' + self.short_classifier[self.classifiers[i]]
                   for i in range(len(classifiers))])

        plt.ylabel(self.suffix + ' : ' + metric.lower())
        plt.title(self.suffix + ' - ' +
                  self.models[self.model] + ' : ' + metric.lower())
        plt.ylim((0, 1))
        plt.axhline(y=0.5, color='blue', linestyle='--')
        plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        file_name = self.output_path + self.suffix + '_histo_' + metric.lower() + \
            '.png'
        plt.savefig(file_name)
        return None

    def read_data(self, metric='Accuracy'):
        output = {}
        for i in range(len(self.classifiers)):
            _classifier = self.classifiers[i]
            output[_classifier] = []
            visited = []
            for dim in self.dimensions:
                path = self.input_path+self.suffix+'-'+str(dim)
                # print(path)
                files = ComputeFile(input_path=path).build_list_files()
                new_data = self.read_csv(files[0])
                # if not dim in visited:
                line = new_data.loc[new_data['Model'] == _classifier]
                # print(_classifier, '>>>>> ', metric, '------', dim, ' >>> ')
                value = line.at[line.index[0], metric]
                # print(value, ' === ', visited)
                output[_classifier].append(value)
                visited.append(dim)
        values = []
        dimensions = ['dim-'+str(i) for i in range(100, 300, 50)]
        for dim in range(0, len(dimensions)):
            _values = []
            for i in range(0, len(self.classifiers)):
                _values.append(output[self.classifiers[i]][dim])
            values.append(_values)
        return values

    def read_csv(self, file=''):
        df = pd.read_csv(file)
        return df

    def run(self):
        print('Histogram generation started 0%')
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1-score']:
            print('Metric : ', metric)
            data = self.read_data(metric=metric)
            # print(data)
            self.plot_data(data=data, metric=metric)
        print('Histogram generation ended 100%')
        return None


if __name__ == '__main__':
    def arg_manager():
        parser = argparse.ArgumentParser()
        parser.add_argument("--input_path", type=str, default="./data/")
        parser.add_argument("--output_path", type=str, default="./outputs/")
        parser.add_argument("--suffix", type=str, default="anatomy")
        parser.add_argument("--model", type=str, default="r2v")
        return parser.parse_args()
    args = arg_manager()
    Main(input_path=args.input_path,
         output_path=args.output_path, suffix=args.suffix, model=args.model).run()
