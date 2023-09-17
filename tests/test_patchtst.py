import torch

from aiq.dataset import TSDataset
from aiq.models import PatchTSTModel
from aiq.utils.config import config as cfg

if __name__ == '__main__':
    # config
    cfg.from_file('./configs/patchtst_model_reg.yaml')
    print(cfg)

    ts_dataset = TSDataset(data_dir='./data', save_dir='./temp', instruments='csi1000', start_time='2020-01-01',
                           end_time='2023-05-31', feature_names=cfg.dataset.feature_names,
                           label_names=cfg.dataset.label_names, adjust_price=True,
                           seq_len=cfg.model.params.seq_len, pred_len=cfg.model.params.pred_len, training=True)
    patch_tst = PatchTSTModel(model_params=cfg.model.params)
    patch_tst.fit(train_dataset=ts_dataset, val_dataset=ts_dataset)
