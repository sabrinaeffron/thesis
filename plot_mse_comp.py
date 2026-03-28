import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bootstrap

models = [
    "gpt-5",
    "o3-mini",
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-35-turbo-16k",
    "Llama-3.3-70B-Instruct",
    "mistral-small-2503"
]

# plot before swap (final)
mse_vals_f, lo_vals_f, hi_vals_f = [], [], []

for model in models:
    df = pd.read_csv(f"psy360_old/g2p2k_test_results_{model}.csv")
    df["squared_error"] = (df["aRate"] - df["pred_A"] / 100.0) ** 2
    mse = np.nanmean((df["aRate"] - df["pred_A"] / 100.0) ** 2)
    mse_vals_f.append(mse)

    x = df["squared_error"].to_numpy()
    x = x[~np.isnan(x)]

    if len(x) >= 2:
        result = bootstrap((x,), np.mean, confidence_level=0.95, n_resamples=1000, method='percentile')
        lo, hi = result.confidence_interval
    else:
        lo, hi = np.nan, np.nan
    lo_vals_f.append(lo)
    hi_vals_f.append(hi)

folders = [
    ("Qwen-2.5-7B-Instruct",  "outputs/qwen-2.5-7b/test_results_qwen_final_0.csv"),
    ("Gemma-2-9B-Instruct",   "outputs/gemma-2-9b/test_results_gemma_final_0.csv"),
    ("Olmo-3-7B-Instruct",    "outputs/olmo-3-7b/test_results_olmo_final_0.csv"),
    ("Llama-3.1-8B-Instruct", "outputs/llama-3.1-8b/test_results_llama_final_0.csv"),
]

for _, path in folders:
    df = pd.read_csv(path)
    df["squared_error"] = (df["aRate"] - df["pred_A"] / 100.0) ** 2
    mse = np.nanmean((df["aRate"] - df["pred_A"] / 100.0) ** 2)
    mse_vals_f.append(mse)

    x = df["squared_error"].to_numpy()
    x = x[~np.isnan(x)]

    if len(x) >= 2:
        result = bootstrap((x,), np.mean, confidence_level=0.95, n_resamples=1000, method='percentile')
        lo, hi = result.confidence_interval
    else:
        lo, hi = np.nan, np.nan
    lo_vals_f.append(lo)
    hi_vals_f.append(hi)

llm_names_f = models + [name for name, _ in folders]

order = np.argsort(mse_vals_f)
names_sorted_f = [llm_names_f[i] for i in order]
mse_sorted_f = np.array([mse_vals_f[i] for i in order])
lo_sorted_f = np.array([lo_vals_f[i] for i in order], dtype=float)
hi_sorted_f = np.array([hi_vals_f[i] for i in order], dtype=float)

lower_err_f = mse_sorted_f - lo_sorted_f
upper_err_f = hi_sorted_f - mse_sorted_f

xpos_f = np.arange(len(names_sorted_f))


# plot after swap
mse_vals, lo_vals, hi_vals = [], [], []

for model in models:
    df = pd.read_csv(f"psy360/g2p2k_test_results_swap_{model}.csv")
    df["squared_error"] = (df["aRate"] - df["pred_A"] / 100.0) ** 2
    mse = np.nanmean((df["aRate"] - df["pred_A"] / 100.0) ** 2)
    mse_vals.append(mse)

    x = df["squared_error"].to_numpy()
    x = x[~np.isnan(x)]

    if len(x) >= 2:
        result = bootstrap((x,), np.mean, confidence_level=0.95, n_resamples=1000, method='percentile')
        lo, hi = result.confidence_interval
    else:
        lo, hi = np.nan, np.nan
    lo_vals.append(lo)
    hi_vals.append(hi)

folders = [
    ("Qwen-2.5-7B-Instruct",  "outputs/qwen-2.5-7b/test_results_qwen_swap_0.csv"),
    ("Gemma-2-9B-Instruct",   "outputs/gemma-2-9b/test_results_gemma_swap_0.csv"),
    ("Olmo-3-7B-Instruct",    "outputs/olmo-3-7b/test_results_olmo_swap_0.csv"),
    ("Llama-3.1-8B-Instruct", "outputs/llama-3.1-8b/test_results_llama_swap_0.csv"),
]

for _, path in folders:
    df = pd.read_csv(path)
    df["squared_error"] = (df["aRate"] - df["pred_A"] / 100.0) ** 2
    mse = np.nanmean((df["aRate"] - df["pred_A"] / 100.0) ** 2)
    mse_vals.append(mse)

    x = df["squared_error"].to_numpy()
    x = x[~np.isnan(x)]

    if len(x) >= 2:
        result = bootstrap((x,), np.mean, confidence_level=0.95, n_resamples=1000, method='percentile')
        lo, hi = result.confidence_interval
    else:
        lo, hi = np.nan, np.nan
    lo_vals.append(lo)
    hi_vals.append(hi)

llm_names = models + [name for name, _ in folders]

order = np.argsort(mse_vals)
names_sorted = [llm_names[i] for i in order]
mse_sorted = np.array([mse_vals[i] for i in order])
lo_sorted = np.array([lo_vals[i] for i in order], dtype=float)
hi_sorted = np.array([hi_vals[i] for i in order], dtype=float)

lower_err = mse_sorted - lo_sorted
upper_err = hi_sorted - mse_sorted

lower_err = np.where(np.isfinite(lower_err), lower_err, 0.0)
upper_err = np.where(np.isfinite(upper_err), upper_err, 0.0)
has_err = np.isfinite(lo_sorted) & np.isfinite(hi_sorted)

xpos = np.arange(len(names_sorted))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5), sharey=True)

ax1.bar(
    xpos_f,
    mse_sorted_f,
    yerr=np.vstack([lower_err_f, upper_err_f]),
    capsize=3,
    label="LLMs (95% CI)"
)
ax1.set_title('MSE values on original data')
ax1.set_xticks(xpos_f, names_sorted_f, rotation=45, ha="right")
ax1.set_ylabel("Mean Squared Error (MSE)")
ax1.set_xlabel("Model")

ax2.bar(
    xpos,
    mse_sorted,
    yerr=np.vstack([lower_err, upper_err]),
    capsize=3,
    label="LLMs (95% CI)"
)
ax2.set_title('MSE values on swapped data')
ax2.set_xticks(xpos, names_sorted, rotation=45, ha="right")
ax2.tick_params(labelleft=True)
ax2.set_ylabel("Mean Squared Error (MSE)")
ax2.set_xlabel("Model")

plt.tight_layout()
plt.show()

for name, mse in zip(names_sorted, mse_sorted):
    print(f"{name:25s}  MSE = {mse:.4f}")