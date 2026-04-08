import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sp

steps = list(range(100, 6500, 400))
steps.insert(0, 0)
steps.append(6522)

models = ["gemma-2-9b", "qwen-2.5-7b", "olmo-3-7b", "llama-3.1-8b"]
model_base = ["gemma", "qwen", "olmo", "llama"]
seeds = ["1000", "2000", "3000", "4000", "5000"]
total_losses, total_sem, total_mse, total_mse_sem = [], [], [], []

for i in range(len(models)):
    loss_vals, sem_vals, mse_vals, mse_sem_vals = [], [], [], []
    for step in steps:
        nums, mse_nums = [], []
        for s in seeds:
            if step == 0:
                file = "outputs/"+models[i]+"/test_results_"+model_base[i]+"_swap_"+str(step)+".csv"
            else:
                file = "outputs/"+models[i]+"/test_results_"+model_base[i]+"_swap"+s+"_"+str(step)+".csv"
            df = pd.read_csv(file)

            bce = np.nanmean(df["bce"])
            nums.append(bce)

            mse = np.nanmean(df["sq_err"])
            mse_nums.append(mse)

        loss_vals.append(np.mean(nums))
        sem_vals.append(sp.sem(nums))

        mse_vals.append(np.mean(mse_nums))
        mse_sem_vals.append(sp.sem(mse_nums))

    total_mse.append(mse_vals)
    total_mse_sem.append(mse_sem_vals)

    total_losses.append(loss_vals)
    total_sem.append(sem_vals)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))

model_names = ["Gemma-2-9B", "Qwen-2.5-7B", "Olmo-3-7B", "Llama-3.1-8B"]

for i, name in enumerate(model_names):
    y1 = np.array(total_losses[i])
    sem = np.array(total_sem[i])
    
    ax1.plot(steps, y1, label=name)
    ax1.fill_between(steps, y1 - sem, y1 + sem, alpha=0.25)

    y2 = np.array(total_mse[i])
    mse_sem = np.array(total_mse_sem[i])

    ax2.plot(steps, y2, label=name)
    ax2.fill_between(steps, y2 - mse_sem, y2 + mse_sem, alpha=0.25)

ax1.set_ylabel("Binary cross entropy loss")
ax1.set_xlabel("Checkpoint")

ax2.set_ylabel("Mean squared error")
ax2.set_xlabel("Checkpoint")

plt.legend()
plt.tight_layout()
plt.show()
