import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from config import IMAGE_OUT_DIR, MODEL_BUNDLE_FILE, TEST_FILE


def main():
    print("=" * 80)
    print("STEP 5: EXPECTED RESULTS, PROPOSED MODEL, AND XAI")
    print("=" * 80)

    if not MODEL_BUNDLE_FILE.exists():
        raise FileNotFoundError(f"Run step 4 first: {MODEL_BUNDLE_FILE}")

    IMAGE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(MODEL_BUNDLE_FILE)
    model = bundle["model"]
    label_encoder = bundle["label_encoder"]

    test_df = pd.read_csv(TEST_FILE)
    X_test = test_df[bundle["feature_columns"]]
    X_sample = X_test.sample(n=min(1000, len(X_test)), random_state=42)

    print(f"Proposed deployment model: {bundle['model_name']}")
    print(f"Deployment artifact: {MODEL_BUNDLE_FILE}")
    print("Expected labels:", ", ".join(bundle["target_labels"]))

    if not hasattr(model, "feature_importances_"):
        print("The selected model has no tree feature_importances_. SHAP summary is skipped.")
        return

    importances = pd.Series(model.feature_importances_, index=bundle["feature_columns"]).sort_values()
    plt.figure(figsize=(8, 5))
    importances.plot(kind="barh")
    plt.title(f"Feature Importance - {bundle['model_name']}")
    plt.tight_layout()
    fi_path = IMAGE_OUT_DIR / f"FI_{bundle['model_name'].replace(' ', '_')}.png"
    plt.savefig(fi_path, dpi=200)
    plt.close()
    print(f"Saved feature importance: {fi_path}")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    low_idx = int(np.where(label_encoder.classes_ == "Low_Engagement")[0][0])

    if isinstance(shap_values, list):
        target_shap = shap_values[low_idx]
    elif np.asarray(shap_values).ndim == 3:
        target_shap = shap_values[:, :, low_idx]
    else:
        target_shap = shap_values

    shap.summary_plot(target_shap, X_sample, show=False)
    summary_path = IMAGE_OUT_DIR / "SHAP_Summary_Global.png"
    plt.savefig(summary_path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP summary: {summary_path}")

    if hasattr(model, "predict"):
        y_pred = model.predict(X_test)
        low_rows = np.where(y_pred == low_idx)[0]
        if len(low_rows) == 0:
            return

        row_idx = int(low_rows[0])
        single_values = explainer.shap_values(X_test.iloc[[row_idx]])
        if isinstance(single_values, list):
            values = single_values[low_idx][0]
            base_value = explainer.expected_value[low_idx]
        elif np.asarray(single_values).ndim == 3:
            values = single_values[0, :, low_idx]
            base_value = explainer.expected_value[low_idx]
        else:
            values = single_values[0]
            base_value = explainer.expected_value

        explanation = shap.Explanation(
            values=values,
            base_values=base_value,
            data=X_test.iloc[row_idx].values,
            feature_names=bundle["feature_columns"],
        )
        shap.plots.waterfall(explanation, show=False)
        local_path = IMAGE_OUT_DIR / f"SHAP_Local_Student_{row_idx}.png"
        plt.savefig(local_path, dpi=250, bbox_inches="tight")
        plt.close()
        print(f"Saved local SHAP explanation: {local_path}")


if __name__ == "__main__":
    main()
