import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

model = "gemma-2-9b"
# "gemma-2-9b", "qwen-2.5-7b", "olmo-3-7b", "llama-3.1-8b"

if model == "olmo-3-7b":
    model_base = "olmo"
    model_name = "Olmo-3-7B"
elif model == "qwen-2.5-7b":
    model_base = "qwen"
    model_name = "Qwen-2.5-7B"
elif model == "gemma-2-9b":
    model_base = "gemma"
    model_name = "Gemma-2-9B"
elif model == "llama-3.1-8b":
    model_base = "llama"
    model_name = "Llama-3.1-8B"

seed = "1000"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))

before_file = "outputs_ood/"+model+"/logits_test_results_swap3_"+seed+"_0.csv"
before_df = pd.read_csv(before_file)
before = before_df["logits_a"]

after_file = "outputs_ood/"+model+"/logits_test_results_swap3_"+seed+"_6522.csv"
after_df = pd.read_csv(after_file)
after = after_df["logits_a"]

bins = list(np.arange(0.0, 1.1, 0.1))

ax1.hist(before, bins=bins)
ax2.hist(after, bins=bins)

ax1.set_ylim(0, 1000)
ax2.set_ylim(0, 1000)

ax1.set_ylabel("Frequency")
ax1.set_xlabel("Response value")
ax1.set_title("Probability of choosing Option A before training for " + model_name)

ax2.set_ylabel("Frequency")
ax2.set_xlabel("Response value")
ax2.set_title("Probability of choosing Option A after training for " + model_name)

plt.tight_layout()
plt.show()
