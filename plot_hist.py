import pandas as pd
import matplotlib.pyplot as plt

model = "olmo-3-7b"
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

# seeds = ["1000_2", "2000_2", "3000_2"]
seed = "1000"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))

# before_file = "outputs/"+model+"/test_results_"+model_base+"_final_0.csv"
before_file = "outputs/"+model+"/test_results_"+model_base+"_swap_0.csv"
before_df = pd.read_csv(before_file)
before = before_df["pred_A"]

after_file = "outputs/"+model+"/test_results_"+model_base+"_swap"+seed+"_6522.csv"
after_df = pd.read_csv(after_file)
after = after_df["pred_A"]

bins = list(range(0, 101, 10))

ax1.hist(before, bins=bins)
ax2.hist(after, bins=bins)

ax1.set_ylabel("Frequency")
ax1.set_xlabel("Response value")
ax1.set_title("Response frequencies before training for " + model_name)

ax2.set_ylabel("Frequency")
ax2.set_xlabel("Response value")
ax2.set_title("Response frequencies after training for " + model_name)

plt.tight_layout()
plt.show()
