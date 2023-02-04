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
    model = XGBModel(feature_cols=['momentum_1d', 'sobv'], label_col=['label_reg'], model_params=model_params)
    model.fit(train_dataset=train_dataset)
    predict_result = model.predict(dataset=train_dataset)
