import pandas as pd
import matplotlib.pyplot as plt

seeds = ["1000_2", "2000_2", "3000_2"]

fig, (ax1) = plt.subplots(1, 1, figsize=(6,5))

file = "g2p2k/games2p2k_text_swap.csv"
df = pd.read_csv(file)
data = [rate * 100 for rate in df["aRate"]]

bins = list(range(0, 101, 10))

ax1.hist(data, bins=bins)

ax1.set_ylabel("Frequency")
ax1.set_xlabel("Empirical percentage")
ax1.set_title("Empirical percentage frequencies in the updated dataset")

plt.tight_layout()
plt.show()
