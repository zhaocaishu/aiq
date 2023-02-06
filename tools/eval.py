import argparse
import os

import numpy as np
from sklearn.metrics import mean_squared_error

from aiq.dataset import Dataset, Alpha100
from aiq.models import XGBModel
from aiq.utils.config import config as cfg


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a model')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for evaluation')
    parser.add_argument('--data_dir', type=str, help='the data directory')
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
    valid_dataset = Dataset(args.data_dir,
                            start_time=cfg.dataset.segments['valid'][0],
                            end_time=cfg.dataset.segments['valid'][1],
                            handler=Alpha100())

    # train model
    model = XGBModel(feature_cols=cfg.dataset.feature_cols,
                     label_col=cfg.dataset.label_col,
                     model_params=cfg.model.params)
    model.load(os.path.join(args.save_dir, 'model.json'))
    predict_result = model.predict(valid_dataset)

    label_reg = predict_result.to_dataframe()['label_reg'].values
    prediction = predict_result.to_dataframe()['prediction'].values
    print("rmse:", np.sqrt(mean_squared_error(label_reg, prediction)))


if __name__ == '__main__':
    main()
