import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sp

# original LLMs
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

mse_values, lo_vals, hi_vals = [], [], []

for model in models:
    df = pd.read_csv(f"psy360_old/g2p2k_test_results_{model}.csv")
    df["squared_error"] = (df["aRate"] - df["pred_A"] / 100.0) ** 2
    mse = np.nanmean((df["aRate"] - df["pred_A"] / 100.0) ** 2)
    mse_values.append(mse)

    x = df["squared_error"].to_numpy()
    x = x[~np.isnan(x)]

    if len(x) >= 2:
        result = sp.bootstrap((x,), np.mean, confidence_level=0.95, n_resamples=1000, method='percentile')
        lo, hi = result.confidence_interval
    else:
        lo, hi = np.nan, np.nan
    lo_vals.append(lo)
    hi_vals.append(hi)

# untrained LLMs
folders = [
    ("Qwen-2.5-7B-Instruct",  "outputs/qwen-2.5-7b/test_results_qwen_final_0.csv"),
    ("Gemma-2-9B-Instruct",   "outputs/gemma-2-9b/test_results_gemma_final_0.csv"),
    ("Olmo-3-7B-Instruct",    "outputs/olmo-3-7b/test_results_olmo_final_0.csv"),
    ("Llama-3.1-8B-Instruct", "outputs/llama-3.1-8b/test_results_llama_final_0.csv"),
]

for _, path in folders:
    df = pd.read_csv(path)
    df["squared_error"] = (df["aRate"] - df["pred_A"] / 100.0) ** 2
    
    mse = np.nanmean(df["squared_error"])
    mse_values.append(mse)

    x = df["squared_error"].to_numpy()
    x = x[~np.isnan(x)]

    result = sp.bootstrap((x,), np.mean, confidence_level=0.95, n_resamples=1000, method='percentile')
    lo, hi = result.confidence_interval

    lo_vals.append(lo)
    hi_vals.append(hi)

llm_names = models + [name for name, _ in folders]

# models from Zhu's research
cogsci_names = ["Random", "Nash", "Best Cognitive without NNs", "Best Cognitive with NNs", "MLP"]
cogsci_mse   = [0.0875, 0.1625, 0.0181, 0.0096, 0.0073]

all_names = llm_names + cogsci_names 
all_mse = mse_values + cogsci_mse
all_lo = lo_vals + [np.nan]*len(cogsci_names)
all_hi = hi_vals + [np.nan]*len(cogsci_names)

order = np.argsort(all_mse)
names_sorted = [all_names[i] for i in order]
mse_sorted = np.array([all_mse[i] for i in order])
lo_sorted = np.array([all_lo[i] for i in order], dtype=float)
hi_sorted = np.array([all_hi[i] for i in order], dtype=float)

lower_err = mse_sorted - lo_sorted
upper_err = hi_sorted - mse_sorted

lower_err = np.where(np.isfinite(lower_err), lower_err, 0.0)
upper_err = np.where(np.isfinite(upper_err), upper_err, 0.0)
has_err = np.isfinite(lo_sorted) & np.isfinite(hi_sorted)

xpos = np.arange(len(names_sorted))

is_cogsci = np.array([name in set(cogsci_names) for name in names_sorted])
is_llm = ~is_cogsci

plt.figure(figsize=(10, 6))

plt.bar(
    xpos[is_llm],
    mse_sorted[is_llm],
    yerr=np.vstack([lower_err[is_llm], upper_err[is_llm]]),
    capsize=3,
    label="LLMs"
)

plt.bar(
    xpos[is_cogsci],
    mse_sorted[is_cogsci],
    color="red",
    label="CogSci models"
)

plt.xticks(xpos, names_sorted, rotation=45, ha="right")
plt.ylabel("Mean Squared Error (MSE)")
plt.xlabel("Model")
plt.legend(loc="upper left", bbox_to_anchor=(0, 1))
plt.tight_layout()
plt.show()

for name, mse in zip(names_sorted, mse_sorted):
    print(f"{name:25s}  MSE = {mse:.4f}")