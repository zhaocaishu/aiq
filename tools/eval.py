import argparse
import os

import torch
import numpy as np
from sklearn.metrics import mean_squared_error

from aiq.dataset import Dataset, TSDataset, Alpha158, Alpha101, ts_split
from aiq.models import XGBModel, LGBModel, DEnsembleModel, PatchTSTModel, NLinearModel
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
    if cfg.model.name in ['PatchTST', 'NLinear']:
        test_dataset = TSDataset(data_dir=args.data_dir, save_dir=args.save_dir, instruments=args.instruments,
                                 start_time=cfg.dataset.segments['test'][0],
                                 end_time=cfg.dataset.segments['test'][1], feature_names=cfg.dataset.feature_names,
                                 label_names=cfg.dataset.label_names, adjust_price=True,
                                 seq_len=cfg.model.params.seq_len, pred_len=cfg.model.params.pred_len, training=False)
    else:
        handlers = (Alpha158(), Alpha101())
        dataset = Dataset(args.data_dir,
                          instruments=args.instruments,
                          start_time=cfg.dataset.start_time,
                          end_time=cfg.dataset.end_time,
                          handlers=handlers)
        test_dataset = ts_split(dataset, [cfg.dataset.segments['test']])[0]
    print('Loaded %d items to test dataset' % len(test_dataset))

    # model
    if cfg.model.name == 'XGB':
        model = XGBModel()
    elif cfg.model.name == 'LGB':
        model = LGBModel()
    elif cfg.model.name == 'DoubleEnsemble':
        model = DEnsembleModel()
    elif cfg.model.name == 'PatchTST':
        model = PatchTSTModel(model_params=cfg.model.params)
    elif cfg.model.name == 'NLinear':
        model = NLinearModel(model_params=cfg.model.params)
    model.load(args.save_dir)

    # evaluation
    mse = model.eval(test_dataset, criterion=torch.nn.MSELoss())
    print("RMSE:", np.sqrt(mse))


if __name__ == '__main__':
    main()
