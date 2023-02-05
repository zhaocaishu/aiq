from aiq.dataset import Dataset, Alpha100
from aiq.models import XGBModel

if __name__ == '__main__':
    train_dataset = Dataset('./data', start_time='2021-08-30', end_time='2022-04-28', handler=Alpha100(), shuffle=True)
    valid_dataset = Dataset('./data', start_time='2022-04-29', end_time='2022-08-26', handler=Alpha100())
    model_params = {
        'objective': 'reg:squarederror',
        'eta': 0.0421,
        'colsample_bytree': 0.8879,
        'max_depth': 8,
        'n_estimators': 647,
        'subsample': 0.8789,
        'nthread': 2,
        'eval_metric': 'rmse'
    }
    model = XGBModel(feature_cols=['momentum_1d', 'momentum_3d', 'momentum_5d', 'momentum_15d', 'momentum_30d',
                                   'highlow_1d', 'highlow_3d', 'highlow_5d', 'highlow_15d', 'highlow_30d',
                                   'vstd_1d', 'vstd_3d', 'vstd_5d', 'vstd_15d', 'vstd_30d', 'sobv', 'rsi', 'macd'],
                     label_col=['label_reg'], model_params=model_params)
    model.fit(train_dataset=train_dataset, val_dataset=valid_dataset)
    result_dataset = model.predict(dataset=valid_dataset)
    result_dataset.dump('./data/prediction_result')
