import numpy as np
import tensorflow as tf
import keras.backend as K
import shap
import matplotlib.pyplot as plt
import os


def get_ig_attributions(model, baseline, input_sample, steps=50):
    """
    compute integrated gradients for one input sample

    this approximates how much each feature contributes
    to the model prediction by integrating gradients
    from a baseline to the actual input

    args:
        model: trained keras model
        baseline: reference input (e.g. mean of dataset)
        input_sample: single sample to explain (shape: 1 x features)
        steps: number of interpolation steps

    returns:
        attributions: contribution score per feature
    """

    # create interpolation points between baseline and input
    alphas = np.linspace(0, 1, steps + 1)[:, np.newaxis]
    delta = input_sample - baseline
    interpolated = baseline + alphas * delta

    # get gradient of output w.r.t input
    # this works even if eager execution is off
    weights = model.output
    gradients = K.gradients(weights, model.input)[0]
    get_gradients = K.function([model.input], [gradients])

    # compute gradients for all interpolated samples
    grads = get_gradients([interpolated])[0]

    # average gradients across steps
    avg_grads = np.mean(grads, axis=0)

    # final IG formula
    attributions = delta * avg_grads

    return attributions


def run_xai(model, X_background, X_explain, feature_names=None):
    """
    runs shap and integrated gradients on a dataset

    also creates plots and saves them to disk

    args:
        model: trained model
        X_background: background data for shap (used as reference)
        X_explain: samples to explain
        feature_names: optional list of feature names
    """

    # create output folder if it does not exist
    os.makedirs("xai_outputs", exist_ok=True)

    if feature_names is None:
        feature_names = [f"Feat_{i}" for i in range(X_background.shape[1])]

    print("\n--- running shap ---")

    # shap sometimes expects input as list for keras models
    explainer = shap.GradientExplainer(model, [X_background])
    shap_values = explainer.shap_values(X_explain)

    # shap may return a list depending on model output
    shap_vals = np.array(shap_values[0]) if isinstance(shap_values, list) else np.array(shap_values)

    print("--- running integrated gradients ---")

    # use mean of background as baseline
    baseline = np.mean(X_background, axis=0, keepdims=True)

    ig_list = []

    # compute IG for each sample separately
    for i in range(len(X_explain)):
        sample = X_explain[i:i+1]
        attr = get_ig_attributions(model, baseline, sample)
        ig_list.append(attr)
    ig_attrs = np.vstack(ig_list)

    # ---------------- plotting ----------------

    # remove extreme shap outliers (helps visualization)
    p_low, p_high = np.percentile(shap_vals, [0.05, 99.95])
    shap_vals_clipped = np.clip(shap_vals, p_low, p_high)

    # global IG importance plot
    plt.figure(figsize=(12, 8))

    mean_abs_ig = np.mean(np.abs(ig_attrs), axis=0)

    # take top 15 features
    top_idx = np.argsort(mean_abs_ig)[-15:]

    labels = [feature_names[i] for i in top_idx]
    values = mean_abs_ig[top_idx]

    plt.barh(labels, values, color="steelblue", edgecolor='black', alpha=0.8)
    plt.xlabel("Importance (Mean Absolute Attribution)")
    plt.title("Integrated Gradients: Top 15 Biological/Chemical Drivers")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("xai_outputs/ig_global.png", dpi=150)
    plt.close()

    # shap summary plot
    shap.summary_plot(shap_vals_clipped, X_explain, feature_names=feature_names, 
                      max_display=15, show=False, plot_size=(12, 8))
    plt.title("SHAP Feature Impact (Clipped 0.5-99.5%)")
    plt.tight_layout()
    plt.savefig("xai_outputs/shap_summary.png", dpi=150)
    plt.close()

    # evaluate faithfulness
    shap_faithfulness = calculate_faithfulness(model, X_explain, shap_vals)
    ig_faithfulness = calculate_faithfulness(model, X_explain, ig_attrs)

    print(f"\nshap faithfulness: {shap_faithfulness:.4f}")
    print(f"ig faithfulness: {ig_faithfulness:.4f}")

    print("\nxai done, check folder")


def calculate_faithfulness(model, X_sample, attributions, top_p=0.1):
    """
    simple faithfulness test

    idea:
    remove the most important features and see how much prediction drops

    args:
        model: trained model
        X_sample: input samples
        attributions: feature importance scores
        top_p: fraction of features to mask

    returns:
        average prediction change
    """

    # original predictions
    orig_pred = model.predict(X_sample)

    num_mask = int(X_sample.shape[1] * top_p)
    X_masked = X_sample.copy()

    for i in range(len(X_sample)):
        # get indices of most important features
        top_indices = np.argsort(np.abs(attributions[i]))[-num_mask:]

        # set them to zero (simple masking)
        X_masked[i, top_indices] = 0

    # predictions after masking
    masked_pred = model.predict(X_masked)

    # measure how much predictions changed
    score = np.mean(np.abs(orig_pred - masked_pred))

    return score