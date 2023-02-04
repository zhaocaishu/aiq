from aiq.dataset import Dataset, Alpha100
from aiq.models import XGBModel

if __name__ == '__main__':
    train_dataset = Dataset('./data/features', symbols=['BABA', 'AAPL'], handler=Alpha100())
    model_params = {
        'objective': 'reg:absoluteerror',
        'learning_rate': 0.1,
        'colsample_bytree': 0.3,
        'max_depth': 10,
        'n_estimators': 100
    }
    model = XGBModel(feature_cols=['momentum_1d', 'momentum_3d', 'momentum_5d', 'momentum_15d', 'momentum_30d',
                                   'highlow_1d', 'highlow_3d', 'highlow_5d', 'highlow_15d', 'highlow_30d',
                                   'vstd_1d', 'vstd_3d', 'vstd_5d', 'vstd_15d', 'vstd_30d',
                                   'sobv', 'rsi', 'macd'],
                     label_col=['label_reg'], model_params=model_params)
    model.fit(train_dataset=train_dataset)
    predict_result = model.predict(dataset=train_dataset)
