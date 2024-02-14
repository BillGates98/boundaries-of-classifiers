import os
from compute_files import ComputeFile
import time
import argparse
import os
import pandas as pd
import numpy as np
import lazypredict
from lazypredict.Supervised import LazyClassifier, CLASSIFIERS
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from poleClassifier import PolClassifier
from architecture import Architecture


class Main:

    def __init__(self, input_path='', output_path='', suffix=''):
        self.input_path = input_path + suffix + '/feature_vector/'
        self.output_path = output_path + suffix + '/'
        self.suffix = suffix
        files = ComputeFile(input_path=self.input_path).build_list_files()
        test_file = self.filter(keyword='test', all=files)
        self.test_data = self.read_csv(test_file)
        print(test_file)
        train_file = self.filter(keyword='train', all=files)
        self.train_data = self.read_csv(train_file)
        self.measure_file = self.output_path + 'measure.csv'
        self.feature_columns = ['source_id', 'target_id', 'pair_id', 'label']
        self.to_ignore = ["LabelPropagation",
                          "LabelSpreading",    "LinearDiscriminantAnalysis"]
        self.classifiers = ["AdaBoostClassifier",    "BaggingClassifier",    "BernoulliNB",    "CalibratedClassifierCV",    "DecisionTreeClassifier",    "DummyClassifier",    "ExtraTreeClassifier",    "ExtraTreesClassifier",    "GaussianNB",    "KNeighborsClassifier", "LinearSVC",
                            "LogisticRegression",    "NearestCentroid",    "NuSVC",    "PassiveAggressiveClassifier",    "Perceptron",    "QuadraticDiscriminantAnalysis",    "RandomForestClassifier",    "RidgeClassifier",    "RidgeClassifierCV",    "SGDClassifier",    "SVC",    "XGBClassifier",    "LGBMClassifier"]
        self.start_time = time.time()

    def filter(self, keyword='', all=[]):
        return [file for file in all if keyword in file][0]

    def read_csv(self, file=''):
        df = pd.read_csv(file)
        df['label'] = df['label'].replace({True: 1, False: 0})
        return df

    def save_to_csv(self, output={}):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        return None

    def append_rows_to_csv(self, new_rows):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        try:
            df = pd.read_csv(self.measure_file)
        except FileNotFoundError:
            df = pd.DataFrame(
                columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-score'])

        new_data = pd.DataFrame(
            new_rows, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-score'])
        df = pd.concat([df, new_data], ignore_index=True)
        df.to_csv(self.measure_file, index=False)

    def metrics(self, model='', y_true=[], y_pred=[]):
        a = round(accuracy_score(y_true, y_pred), 2)
        p = round(precision_score(y_true, y_pred), 2)
        r = round(recall_score(y_true, y_pred), 2)
        f = round(f1_score(y_true, y_pred), 2)
        self.append_rows_to_csv([(model, a, p, r, f)])

    def select_classifiers(self):
        output = []
        for name, model in CLASSIFIERS:
            if not name in self.to_ignore:
                output.append((name, model))
        return output

    def run(self):
        labels = self.feature_columns
        test = self.test_data
        y_test = np.array(test['label'].tolist())
        X_test = np.array(test.drop(labels, axis=1))
        train = self.train_data
        y_train = np.array(train['label'].tolist())
        X_train = np.array(train.drop(labels, axis=1))

        # clf = PolClassifier(X_train, X_test, y_train, y_test)
        # predictions = clf.fit()

        clf = Architecture(X_train, X_test, y_train, y_test)
        predictions = clf.fit()
        # if os.path.exists(self.measure_file):
        #     os.remove(self.measure_file)
        # for model in predictions:
        #     self.metrics(model=model, y_true=y_test, y_pred=predictions[model])
        return None


if __name__ == '__main__':

    def arg_manager():
        parser = argparse.ArgumentParser()
        parser.add_argument("--input_path", type=str, default="./data/")
        parser.add_argument("--output_path", type=str, default="./outputs/")
        parser.add_argument("--suffix", type=str, default="anatomy-20")
        return parser.parse_args()
    args = arg_manager()
    Main(input_path=args.input_path,
         output_path=args.output_path, suffix=args.suffix).run()
