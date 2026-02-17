from data_generation import plot_sample_initial_curves
from data_generation import generate_dataset
from collinearity import iterative_vif_pruning
from train_model import train_xgb
from evaluate import evaluate_model, plot_predictions, tail_error
from feature_analysis import analyze_feature_importance

def main():

    print("Plotting sample initial curves...")
    plot_sample_initial_curves(n_curves=5)

    print("Generating dataset...")
    df = generate_dataset()

    df = iterative_vif_pruning(df)

    print("\nTraining model...")
    model, X_test, y_test = train_xgb(df)

    print("\nEvaluating model...")
    preds = evaluate_model(model, X_test, y_test)
    plot_predictions(y_test, preds)
    tail_error(y_test, preds)

    print("\nFeature importance...")
    analyze_feature_importance(model)

if __name__ == "__main__":
    main()
