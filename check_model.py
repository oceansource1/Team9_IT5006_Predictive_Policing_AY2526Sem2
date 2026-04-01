import joblib

xgb_obj = joblib.load("models/xgb_grid.pkl")
print(type(xgb_obj))

if isinstance(xgb_obj, dict):
    print(xgb_obj.keys())
    for k, v in xgb_obj.items():
        print(k, type(v))