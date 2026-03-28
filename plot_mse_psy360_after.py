import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sp

# original models
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

folders = [
    ("Qwen-2.5-7B-Instruct (original)",  "outputs/qwen-2.5-7b/test_results_qwen_final_0.csv"),
    ("Gemma-2-9B-Instruct (original)",   "outputs/gemma-2-9b/test_results_gemma_final_0.csv"),
    ("Olmo-3-7B-Instruct (original)",    "outputs/olmo-3-7b/test_results_olmo_final_0.csv"),
    ("Llama-3.1-8B-Instruct (original)", "outputs/llama-3.1-8b/test_results_llama_final_0.csv"),
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

# fine-tuned models
models_train = ["gemma-2-9b", "qwen-2.5-7b", "olmo-3-7b", "llama-3.1-8b"]
model_base_train = ["gemma", "qwen", "olmo", "llama"]
seeds = ["1000", "2000", "3000", "4000", "5000"]
llm_names_train = ["Gemma-2-9B-Instruct (fine-tuned)", 
                   "Qwen-2.5-7B-Instruct (fine-tuned)", 
                   "Olmo-3-7B-Instruct (fine-tuned)", 
                   "Llama-3.1-8B-Instruct (fine-tuned)"]

for i in range(len(models_train)):
    mse_nums, lo_nums, hi_nums = [], [], []
    for s in seeds:
        file = "outputs/"+models_train[i]+"/test_results_"+model_base_train[i]+"_swap"+s+"_6522.csv"
        df = pd.read_csv(file)

        mse = np.nanmean(df["sq_err"])
        mse_nums.append(mse)

        x = df["sq_err"].to_numpy()
        x = x[~np.isnan(x)]

        result = sp.bootstrap((x,), np.mean, confidence_level=0.95, n_resamples=1000, method='percentile')
        lo, hi = result.confidence_interval

        lo_nums.append(lo)
        hi_nums.append(hi)

    mse_values.append(np.mean(mse_nums))
    lo_vals.append(np.nanmean(lo_nums))
    hi_vals.append(np.nanmean(hi_nums))

# models from Zhu's research
cogsci_names = ["Random", "Nash", "Best Cognitive", "MLP"]
cogsci_mse   = [0.0875, 0.1625, 0.0096, 0.0073]

all_names = llm_names +llm_names_train + cogsci_names 
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

is_trained = np.array([name in set(llm_names_train) for name in names_sorted])
is_cogsci = np.array([name in set(cogsci_names) for name in names_sorted])
is_untrained = ~is_trained & ~is_cogsci

plt.figure(figsize=(10, 6))

plt.bar(
    xpos[is_untrained],
    mse_sorted[is_untrained],
    yerr=np.vstack([lower_err[is_untrained], upper_err[is_untrained]]),
    capsize=3,
    label="LLMs"
)

plt.bar(
    xpos[is_trained],
    mse_sorted[is_trained],
    yerr=np.vstack([lower_err[is_trained], upper_err[is_trained]]),
    capsize=3,
    color="green",
    label="Fine-tuned LLMs"
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