
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

def compute_vif(df, target="mpe_normalized"):
    X = df.drop(columns=[target])
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X.values, i)
        for i in range(X.shape[1])
    ]
    return vif_data.sort_values("VIF", ascending=False)

def iterative_vif_pruning(df, threshold=10):
    print("\nStarting VIF pruning...")
    while True:
        vif = compute_vif(df)
        max_vif = vif["VIF"].max()

        if max_vif < threshold:
            break

        drop_feature = vif.iloc[0]["feature"]
        print(f"Dropping '{drop_feature}' with VIF={max_vif:.2f}")
        df = df.drop(columns=[drop_feature])

    print("\nFinal VIF Table:")
    final_vif = compute_vif(df)
    print(final_vif)

    plt.figure(figsize=(8, 5))
    plt.barh(final_vif["feature"], final_vif["VIF"])
    plt.axvline(x=5, linestyle="--")
    plt.axvline(x=10, linestyle="--")
    plt.xlabel("VIF")
    plt.title("Final VIF After Pruning")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    return df
