
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rel_mae = np.mean(np.abs(y_test - preds)) / np.mean(y_test)
    r2 = r2_score(y_test, preds)
    bias = np.mean(preds - y_test)

    print(f"MAE: {mae:.6f}")
    print(f"Relative MAE: {rel_mae:.2%}")
    print(f"R2: {r2:.4f}")
    print(f"Mean Bias: {bias:.6f}")

    return preds

def plot_predictions(y_test, preds):
    plt.figure()
    plt.scatter(y_test, preds, alpha=0.4)
    plt.xlabel("True MPE")
    plt.ylabel("Predicted MPE")
    plt.title("Predicted vs True MPE")
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()])
    plt.show()

def tail_error(y_test, preds, percentile=95):
    threshold = np.percentile(y_test, percentile)
    mask = y_test >= threshold
    tail_rel_error = np.mean(np.abs(y_test[mask] - preds[mask])) / np.mean(y_test[mask])
    print(f"Tail Relative Error ({percentile}th+): {tail_rel_error:.2%}")
