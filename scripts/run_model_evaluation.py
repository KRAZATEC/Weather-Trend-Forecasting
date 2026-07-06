"""
run_model_evaluation.py
========================
End-to-end: trains all models, evaluates them, and generates the
prediction/residual/feature-importance/model-comparison figures
(Phases 6 & 9 deliverables).
"""

from pathlib import Path
import sys

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, visualization as viz
from src.advanced_analytics import compute_permutation_importance, compute_shap_values
from src.evaluate import compute_metrics, cross_validate_model, evaluate_models, get_best_model
from src.train import get_feature_matrix, get_model_registry, train_all_models, train_test_split_data
from src.utils import get_logger

logger = get_logger(__name__)


def main():
    df = pd.read_csv(config.FEATURED_DATA_FILE, parse_dates=[config.DATETIME_COL])
    X, y, feature_cols = get_feature_matrix(df)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    fitted = train_all_models(X_train, y_train)
    results = evaluate_models(fitted, X_test, y_test)
    results.to_csv(config.REPORTS_DIR / "model_comparison.csv", index=False)

    # Prediction vs actual + residuals for every model
    for name, model in fitted.items():
        preds = model.predict(X_test)
        viz.plot_prediction_vs_actual(y_test, preds, name)
        viz.plot_residuals(y_test, preds, name)

    # Model comparison bar chart
    viz.plot_model_comparison(results, metric="RMSE")
    viz.plot_model_comparison(results, metric="MAE", name="model_comparison_mae")

    # Feature importance for tree-based models
    for name in ["RandomForest", "XGBoost", "LightGBM", "GradientBoosting"]:
        if name in fitted:
            model = fitted[name]
            importances = pd.Series(model.feature_importances_, index=feature_cols)
            viz.plot_feature_importance(importances, name)

    # Permutation importance (best model)
    best_name, best_model = get_best_model(results, fitted)
    logger.info("Best model: %s", best_name)
    perm_imp = compute_permutation_importance(best_model, X_test, y_test, n_repeats=5)
    viz.plot_feature_importance(perm_imp, f"{best_name}_permutation", name="permutation_importance_best_model")

    # SHAP (only for tree models, gracefully skipped if shap missing)
    if best_name in ("RandomForest", "XGBoost", "LightGBM", "GradientBoosting", "DecisionTree"):
        shap_values, sample = compute_shap_values(best_model, X_test)
        if shap_values is not None:
            import shap
            import matplotlib.pyplot as plt
            shap.summary_plot(shap_values, sample, show=False)
            plt.gcf().savefig(config.FIGURES_DIR / "shap_summary.png", bbox_inches="tight")
            plt.close()
            logger.info("Saved SHAP summary plot")

    # 5-fold cross-validation on the best model
    cv_result = cross_validate_model(best_model, X, y)
    logger.info("Cross-validation RMSE (5-fold) for %s: mean=%.3f std=%.3f",
                best_name, cv_result["mean"], cv_result["std"])

    pd.Series(cv_result).to_json(config.REPORTS_DIR / "cv_result.json")
    joblib.dump({"feature_cols": feature_cols, "best_model": best_name}, config.MODELS_DIR / "feature_cols.pkl")

    print(results.round(3).to_string(index=False))
    print(f"\nBest model: {best_name}")
    print(f"5-fold CV RMSE: {cv_result['mean']:.3f} +/- {cv_result['std']:.3f}")


if __name__ == "__main__":
    main()
