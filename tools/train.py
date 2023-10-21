import argparse
import os

from aiq.dataset import Dataset, TSDataset, Alpha158, Alpha101, ts_split
from aiq.models import XGBModel, LGBModel, DEnsembleModel, PatchTSTModel, NLinearModel
from aiq.utils.config import config as cfg


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--data_dir', type=str, help='the data directory')
    parser.add_argument('--instruments', type=str, default='csi1000', help='instruments name')
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
        train_dataset = TSDataset(data_dir=args.data_dir, save_dir=args.save_dir, instruments=args.instruments,
                                  start_time=cfg.dataset.segments['train'][0],
                                  end_time=cfg.dataset.segments['train'][1], feature_names=cfg.dataset.feature_names,
                                  label_names=cfg.dataset.label_names, adjust_price=True,
                                  seq_len=cfg.model.params.seq_len, pred_len=cfg.model.params.pred_len, training=True)
        val_dataset = TSDataset(data_dir=args.data_dir, save_dir=args.save_dir, instruments=args.instruments,
                                start_time=cfg.dataset.segments['valid'][0],
                                end_time=cfg.dataset.segments['valid'][1], feature_names=cfg.dataset.feature_names,
                                label_names=cfg.dataset.label_names, adjust_price=True,
                                seq_len=cfg.model.params.seq_len, pred_len=cfg.model.params.pred_len, training=False)
    else:
        handlers = (Alpha158(), Alpha101())
        dataset = Dataset(args.data_dir,
                          instruments=args.instruments,
                          start_time=cfg.dataset.start_time,
                          end_time=cfg.dataset.end_time,
                          handlers=handlers)
        train_dataset, val_dataset = ts_split(dataset=dataset,
                                              segments=[cfg.dataset.segments['train'], cfg.dataset.segments['valid']])
    print('Loaded %d items to train dataset, %d items to validation dataset' % (len(train_dataset), len(val_dataset)))

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
    elif cfg.model.name == 'PatchTST':
        model = PatchTSTModel(model_params=cfg.model.params)
    elif cfg.model.name == 'NLinear':
        model = NLinearModel(model_params=cfg.model.params)

    model.fit(train_dataset=train_dataset, val_dataset=val_dataset)

    # save processor and model
    model.save(model_dir=args.save_dir)

    print('Model training has been finished successfully!')


if __name__ == '__main__':
    main()
