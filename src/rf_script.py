from sklearn.ensemble import RandomForestRegressor

def model_fn(model_dir):
    model = RandomForestRegressor()
    return model