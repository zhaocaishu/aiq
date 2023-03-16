import argparse
import os

from aiq.dataset import Dataset, Alpha158
from aiq.models import XGBModel, LGBModel, DEnsembleModel
from aiq.utils.config import config as cfg


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
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
    train_dataset = Dataset(args.data_dir,
                            instruments=args.instruments,
                            start_time=cfg.dataset.segments['train'][0],
                            end_time=cfg.dataset.segments['train'][1],
                            handler=handler,
                            training=True)
    valid_dataset = Dataset(args.data_dir,
                            instruments=args.instruments,
                            start_time=cfg.dataset.segments['valid'][0],
                            end_time=cfg.dataset.segments['valid'][1],
                            handler=handler,
                            processor=train_dataset.processor)
    print('Loaded %d items to train dataset, %d items to validation dataset' % (len(train_dataset), len(valid_dataset)))

    # train model
    if cfg.model.name == 'XGB':
        model = XGBModel(feature_cols=train_dataset.feature_names,
                         label_col=train_dataset.label_name,
                         model_params=cfg.model.params)
    elif cfg.model.name == 'LGB':
        model = LGBModel(feature_cols=train_dataset.feature_names,
                         label_col=train_dataset.label_name,
                         model_params=dict(cfg.model.params))
    elif cfg.model.name == 'DoubleEnsemble':
        model = DEnsembleModel(feature_cols=train_dataset.feature_names,
                               label_col=[train_dataset.label_name],
                               **dict(cfg.model.params))

    model.fit(train_dataset=train_dataset, val_dataset=valid_dataset)

    # save model
    model.save(model_dir=args.save_dir)

    print('Model training has been finished successfully!')


if __name__ == '__main__':
    main()
