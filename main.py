import os
from compute_files import ComputeFile
import time
import argparse
import os
import pandas as pd

class Main: 

    def __init__(self, input_path='', output_path='', suffix=''):
        self.input_path = input_path + suffix + '/feature_vector/'
        self.output_path = output_path + suffix + '/'
        self.suffix = suffix
        files = ComputeFile(input_path=self.input_path).build_list_files()
        test_file = self.filter(keyword='test', all=files)
        self.test_data = self.read_csv(test_file)
        train_file = self.filter(keyword='train', all=files)
        self.train_data = self.read_csv(train_file)
        self.feature_columns = ['source_id', 'target_id', 'pair_id', 'label']
        self.start_time = time.time()
    
    def filter(self, keyword='', all=[]):
        return [file for file in all if keyword in file][0]
    
    def read_csv(self, file=''):
        df = pd.read_csv(file, index_col=False)
        df.loc[df["label"] == True] = 1
        df.loc[df["label"] == False] = 0
        return df
    
    def save_to_csv(self, output={}):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        # prefix_file = output_path + ''
        # df = pd.DataFrame(output)
        return None

    def run(self):
        test = self.test_data
        train = self.train_data
        self.save_to_csv(output={})
        return None


if __name__ == '__main__' :
    def arg_manager():
        parser = argparse.ArgumentParser()
        parser.add_argument("--input_path", type=str, default="./data/")
        parser.add_argument("--output_path", type=str, default="./outputs/")
        parser.add_argument("--suffix", type=str, default="abt-buy")
        return parser.parse_args()
    args = arg_manager()
    Main(input_path=args.input_path, output_path=args.output_path, suffix=args.suffix).run()