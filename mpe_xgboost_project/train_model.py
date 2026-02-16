
import xgboost as xgb
from sklearn.model_selection import train_test_split

def train_xgb(df):
    X = df.drop("mpe_normalized", axis=1)
    y = df["mpe_normalized"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = xgb.XGBRegressor(
        n_estimators=2000,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        early_stopping_rounds=50
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    print("Best iteration:", model.best_iteration)

    return model, X_test, y_test
