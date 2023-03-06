from aiq.dataset import Dataset, Alpha158
from aiq.models import DEnsembleModel

if __name__ == '__main__':
    handler = Alpha158()
    train_dataset = Dataset('./data', instruments='all', start_time='2021-08-30', end_time='2022-04-28',
                            handler=handler, adjust_price=False, training=True)
    valid_dataset = Dataset('./data', instruments='all', start_time='2022-04-29', end_time='2022-08-26',
                            handler=handler, adjust_price=False)
    model_params = {
        'base_model': 'gbm',
        'loss': 'mse',
        'num_models': 6,
        'enable_sr': True,
        'enable_fs': True,
        'alpha1': 1,
        'alpha2': 1,
        'bins_sr': 10,
        'bins_fs': 5,
        'decay': 0.5,
        'sample_ratios': [0.8, 0.7, 0.6, 0.5, 0.4],
        'sub_weights': [1, 0.2, 0.2, 0.2, 0.2, 0.2],
        'epochs': 28,
        'colsample_bytree': 0.8879,
        'learning_rate': 0.2,
        'subsample': 0.8789,
        'lambda_l1': 205.6999,
        'lambda_l2': 580.9768,
        'max_depth': 8,
        'num_leaves': 210,
        'num_threads': 4,
        'verbosity': -1
    }

    # train stage
    model = DEnsembleModel(feature_cols=train_dataset.feature_names,
                           label_col=[train_dataset.label_name],
                           **model_params)
    model.fit(train_dataset=train_dataset, val_dataset=valid_dataset)
    model.save(model_dir='./temp')

    # predict stage
    model_eval = DEnsembleModel()
    model_eval.load(model_dir='./temp')
    model_eval.predict(dataset=valid_dataset)
