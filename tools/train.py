import argparse
import os

from aiq.dataset import Dataset, Alpha100
from aiq.models import XGBModel
from aiq.utils.config import config as cfg


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
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
    train_dataset = Dataset(args.data_dir,
                            start_time=cfg.dataset.segments['train'][0],
                            end_time=cfg.dataset.segments['train'][1],
                            handler=Alpha100(),
                            shuffle=True)
    valid_dataset = Dataset(args.data_dir,
                            start_time=cfg.dataset.segments['valid'][0],
                            end_time=cfg.dataset.segments['valid'][1],
                            handler=Alpha100())

    # train model
    model = XGBModel(feature_cols=cfg.dataset.feature_cols,
                     label_col=cfg.dataset.label_col,
                     model_params=cfg.model.params)
    model.fit(train_dataset=train_dataset, val_dataset=valid_dataset)

    # save model
    model.save(os.path.join(args.save_dir, 'model.json'))


if __name__ == '__main__':
    main()
