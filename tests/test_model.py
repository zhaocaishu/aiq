from aiq.dataset import Dataset, Alpha100
from aiq.models import XGBModel

if __name__ == '__main__':
    train_dataset = Dataset('./data/features', symbols=['BABA'], handler=Alpha100())
    model_params = {
        'objective': 'reg:squarederror',
        'learning_rate': 0.1,
        'colsample_bytree': 0.3,
        'max_depth': 5,
        'n_estimators': 10
    }
    model = XGBModel(feature_cols=['vstd_1m', 'sobv'], label_col=['label'], model_params=model_params)
    model.fit(train_dataset=train_dataset)
    predict_result = model.predict(dataset=train_dataset)
    print(predict_result.shape, train_dataset.to_dataframe().shape)
