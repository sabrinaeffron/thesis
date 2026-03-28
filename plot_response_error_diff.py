import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sp

models = ["gemma-2-9b", "qwen-2.5-7b", "olmo-3-7b", "llama-3.1-8b"]
seeds = ["1000", "2000", "3000", "4000", "5000"]

# loss_type = "Mean squared error"
# loss_type = "Mean absolute error"
loss_type = "Cross entropy loss"

steps = list(range(100, 6500, 400))
steps.insert(0, 0)
steps.append(6522)

total_nash, total_opps, total_nash_sem, total_opp_sem = [], [], [], []

for model_name in models:
    nash_error, opp_error, nash_sem, opp_sem = [], [], [], []

    for i in steps:
        opp_nums, nash_nums = [], []

        for s in seeds:
            file = "outputs_ood/"+model_name+"/logits_test_results_swap3_"+s+"_"+str(i)+".csv"
            df = pd.read_csv(file)
            filtered = df[df["p_NE_a"] != df["p_BR_to_opp"]]
            print(len(filtered))

            if loss_type == "Mean squared error":
                nash = np.mean((filtered["logits_a"] - filtered["p_NE_a"]) ** 2)
                opp = np.mean((filtered["logits_a"] - filtered["p_BR_to_opp"]) ** 2)
            elif loss_type == "Mean absolute error":
                nash = np.mean(abs(filtered["logits_a"] - filtered["p_NE_a"]))
                opp = np.mean(abs(filtered["logits_a"] - filtered["p_BR_to_opp"]))
            elif loss_type == "Cross entropy loss":
                eps = 1e-12  # avoid log(0)
                p = filtered["logits_a"].clip(eps, 1 - eps)
                y_nash = filtered["p_NE_a"]
                y_opp = filtered["p_BR_to_opp"]
                nash = np.mean(-(y_nash * np.log(p) + (1 - y_nash) * np.log(1 - p)))
                opp = np.mean(-(y_opp * np.log(p) + (1 - y_opp) * np.log(1 - p)))
            
            nash_nums.append(nash)
            opp_nums.append(opp)

        nash_error.append(np.mean(nash_nums))
        opp_error.append(np.mean(opp_nums))

        nash_sem.append(sp.sem(nash_nums))
        opp_sem.append(sp.sem(opp_nums))

    total_nash.append(nash_error)
    total_opps.append(opp_error)
    total_nash_sem.append(nash_sem)
    total_opp_sem.append(opp_sem)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5), sharey=True)
# fig, ax1 = plt.subplots(figsize=(8, 4))

model_names = ["Gemma-2-9B", "Qwen-2.5-7B", "Olmo-3-7B", "Llama-3.1-8B"]

for i, name in enumerate(model_names):
    ax1.plot(steps, total_nash[i], linewidth=2, label=name)
    ax1.fill_between(
        steps, 
        np.array(total_nash[i]) - np.array(total_nash_sem[i]), 
        np.array(total_nash[i]) + np.array(total_nash_sem[i]), 
        alpha=0.25
    )

ax1.set_xlabel("Training step")
ax1.set_ylabel(loss_type)
ax1.set_title("Error Compared to Nash Equilibrium")

for i, name in enumerate(model_names):
    ax2.plot(steps, total_opps[i], linewidth=2, label=name)
    ax2.fill_between(
        steps,
        np.array(total_opps[i]) - np.array(total_opp_sem[i]),
        np.array(total_opps[i]) + np.array(total_opp_sem[i]),
        alpha=0.25
    )

ax2.set_xlabel("Training step")
ax2.set_ylabel(loss_type)
ax2.set_title("Error Compared to Empirical Opponent Actions")

plt.legend(loc="center right", bbox_to_anchor=(1,0.5))
plt.tight_layout()
plt.show()