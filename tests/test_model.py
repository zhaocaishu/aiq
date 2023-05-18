from aiq.dataset import Dataset, Alpha158, Alpha101, ts_split
from aiq.models import XGBModel, LGBModel

if __name__ == '__main__':
    handlers = (Alpha158(), Alpha101())
    dataset = Dataset('./data', instruments='all', start_time='2021-08-30', end_time='2022-08-26',
                      handlers=handlers, adjust_price=False, training=True)
    train_dataset, val_dataset = ts_split(dataset=dataset,
                                          segments=[['2021-08-30', '2022-04-28'], ['2022-04-29', '2022-08-26']])

    model_params = {
        'objective': 'mse',
        'learning_rate': 0.2,
        'colsample_bytree': 0.8879,
        'max_depth': 8,
        'num_leaves': 210,
        'subsample': 0.8789,
        'lambda_l1': 205.6999,
        'lambda_l2': 580.9768,
        'metric': 'rmse',
        'nthread': 4
    }

    # train stage
    model = LGBModel(feature_cols=train_dataset.feature_names,
                     label_col=[train_dataset.label_name],
                     model_params=model_params)
    model.fit(train_dataset=train_dataset, val_dataset=val_dataset)
    model.save(model_dir='./temp')

    # predict stage
    model_eval = LGBModel()
    model_eval.load(model_dir='./temp')
    model_eval.predict(dataset=val_dataset)
