from aiq.dataset import Dataset, Alpha158, ts_split
from aiq.models import XGBModel, LGBModel

if __name__ == '__main__':
    handler = Alpha158()
    dataset = Dataset('./data', instruments='all', start_time='2021-08-30', end_time='2022-08-26',
                      handler=handler, adjust_price=False, training=True)
    train_dataset, val_dataset = ts_split(dataset=dataset,
                                          segments=[['2021-08-30', '2022-04-28'], ['2022-04-29', '2022-08-26']])

    model_params = {
        'objective': 'binary:logistic',
        'eta': 0.0421,
        'tree_method': 'hist',
        'colsample_bytree': 0.8879,
        'max_depth': 8,
        'subsample': 0.8789,
        'reg_alpha': 10,
        'reg_lambda': 10,
        'disable_default_eval_metric': True,
        'nthread': 4
    }

    # train stage
    model = XGBModel(feature_cols=train_dataset.feature_names,
                     label_col=[train_dataset.label_name],
                     model_params=model_params,
                     use_ordinal_reg=True)
    model.fit(train_dataset=train_dataset, val_dataset=val_dataset)
    model.save(model_dir='./temp')

    # predict stage
    model_eval = XGBModel(use_ordinal_reg=True)
    model_eval.load(model_dir='./temp')
    model_eval.predict(dataset=val_dataset)
