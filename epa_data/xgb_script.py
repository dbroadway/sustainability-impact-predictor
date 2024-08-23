import xgboost as xgb

def model_fn(model_dir):
    model = xgb.Booster()
    model.load_model(model_dir)
    return model