from aiq.dataset import Dataset, Alpha158
from aiq.models import XGBModel

if __name__ == '__main__':
    handler = Alpha158()
    train_dataset = Dataset('./data', start_time='2021-08-30', end_time='2022-04-28', handler=handler, shuffle=True)
    valid_dataset = Dataset('./data', start_time='2022-04-29', end_time='2022-08-26', handler=handler)
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
    model = XGBModel(feature_cols=handler.feature_names,
                     label_col=[handler.label_name],
                     model_params=model_params)
    model.fit(train_dataset=train_dataset, val_dataset=valid_dataset)
    result_dataset = model.predict(dataset=valid_dataset)
    result_dataset.dump('./data/prediction_result')
