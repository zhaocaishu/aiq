import torch

from aiq.dataset import TSDataset
from aiq.models import PatchTSTModel
from aiq.utils.config import config as cfg


if __name__ == '__main__':
    # config
    cfg.from_file('./configs/patchtst_model_reg.yaml')
    print(cfg)

    train_dataset = TSDataset(data_dir='./data', save_dir='./temp', instruments='csi1000', start_time=cfg.dataset.segments['train'][0],
                              end_time=cfg.dataset.segments['train'][1], feature_names=cfg.dataset.feature_names,
                              label_names=cfg.dataset.label_names, adjust_price=True,
                              seq_len=cfg.model.params.seq_len, pred_len=cfg.model.params.pred_len, training=True)
    val_dataset = TSDataset(data_dir='./data', save_dir='./temp', instruments='csi1000', start_time=cfg.dataset.segments['valid'][0],
                            end_time=cfg.dataset.segments['valid'][1], feature_names=cfg.dataset.feature_names,
                            label_names=cfg.dataset.label_names, adjust_price=True,
                            seq_len=cfg.model.params.seq_len, pred_len=cfg.model.params.pred_len, training=True)
    patch_tst = PatchTSTModel(model_params=cfg.model.params)
    patch_tst.fit(train_dataset=train_dataset, val_dataset=val_dataset)
    patch_tst.save(model_dir='./temp')
    patch_tst.load(model_dir='./temp')
