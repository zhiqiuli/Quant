
import xgboost as xgb
import matplotlib.pyplot as plt

def analyze_feature_importance(model):
    importance = model.get_booster().get_score(importance_type='gain')
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    print("Feature Importance (by gain):")
    for feat, val in sorted_imp:
        print(f"{feat}: {val:.4f}")

    xgb.plot_importance(model)
    plt.title("Feature Importance")
    plt.show()
