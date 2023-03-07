import argparse
import os

import numpy as np
from sklearn.metrics import mean_squared_error

from aiq.dataset import Dataset, Alpha158, random_split
from aiq.models import XGBModel, LGBModel
from aiq.utils.config import config as cfg


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a model')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for evaluation')
    parser.add_argument('--data_dir', type=str, help='the data directory')
    parser.add_argument('--instruments', type=str, default='all', help='instruments name')
    parser.add_argument('--save_dir', type=str, help='the saved directory')

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
                      start_time=cfg.dataset.segments['train'][0],
                      end_time=cfg.dataset.segments['test'][1],
                      handler=handler)
    test_dataset = random_split(dataset, [cfg.dataset.segments['test']])[0]
    print('Loaded %d items to test dataset' % len(test_dataset))

    # evaluation
    if cfg.model.name == 'XGB':
        model = XGBModel()
    elif cfg.model.name == 'LGB':
        model = LGBModel()
    elif cfg.model.name == 'DoubleEnsemble':
        model = DEnsembleModel()
    model.load(args.save_dir)
    df_prediction = model.predict(test_dataset).to_dataframe()

    label_reg = df_prediction[dataset.label_name].values
    prediction = df_prediction['PREDICTION'].values
    print("RMSE:", np.sqrt(mean_squared_error(label_reg, prediction)))


if __name__ == '__main__':
    main()
