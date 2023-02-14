import argparse
import os

import numpy as np
from sklearn.metrics import mean_squared_error

from aiq.dataset import Dataset, Alpha158
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
    valid_dataset = Dataset(args.data_dir,
                            instruments=args.instruments,
                            start_time=cfg.dataset.segments['valid'][0],
                            end_time=cfg.dataset.segments['valid'][1],
                            handler=handler)

    # evaluation
    if cfg.model.name == 'XGB':
        model = XGBModel(feature_cols=handler.feature_names,
                         label_col=handler.label_name,
                         model_params=cfg.model.params)
    elif cfg.model.name == 'LGB':
        model = LGBModel(feature_cols=handler.feature_names,
                         label_col=handler.label_name,
                         model_params=cfg.model.params)
    model.load(os.path.join(args.save_dir, 'model.json'))
    df_prediction = model.predict(valid_dataset).to_dataframe()

    label_reg = df_prediction[handler.label_name].values
    prediction = df_prediction['PREDICTION'].values
    print("RMSE:", np.sqrt(mean_squared_error(label_reg, prediction)))


if __name__ == '__main__':
    main()
