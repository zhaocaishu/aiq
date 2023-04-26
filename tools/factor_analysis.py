import argparse
import os
import pickle

import numpy as np
from sklearn.metrics import mean_squared_error

from aiq.dataset import Dataset, Alpha158, ts_split
from aiq.models import XGBModel, LGBModel, DEnsembleModel
from aiq.utils.config import config as cfg
from aiq.evaluation import IC


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a model')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for evaluation')
    parser.add_argument('--data_dir', type=str, help='the data directory')
    parser.add_argument('--instruments', type=str, default='all', help='instruments name')

    args = parser.parse_args()

    return args


def main():
    # parse args
    args = parse_args()

    # config
    cfg.from_file(args.cfg_file)
    print(cfg)

    # dataset
    print(cfg.dataset.segments)
    handler = Alpha158()
    dataset = Dataset(args.data_dir,
                      instruments=args.instruments,
                      start_time=cfg.dataset.start_time,
                      end_time=cfg.dataset.end_time,
                      handler=handler)
    test_dataset = ts_split(dataset, [cfg.dataset.segments['test']])[0]
    test_df = test_dataset.to_dataframe()
    print('Loaded %d items to test dataset' % len(test_dataset))

    eval_results = dict()
    ic_analysis = IC()
    for feature_name in test_dataset.feature_names:
        try:
            eval_result = ic_analysis.eval(test_df, feature_name)
            eval_results[feature_name] = eval_result
        except:
            print('Pass')

    with open('./eval_results.pkl', 'wb') as f:
        pickle.dump(eval_results, f)


if __name__ == '__main__':
    main()
