import pandas as pd
import numpy as np

full = pd.read_csv("g2p2k/games2p2k_text_responses.csv")
test = pd.read_csv("outputs_ood/gemma-2-9b/logits_test_results_full_1600.csv")

cols = ["game_id", "text", "p_NE_a", "p_BR_to_opp", "opp_q"]

df = test.merge(full[cols], on=["game_id", "text"], how="left")

mse_nash = np.nanmean((df["logits_a"] - df["p_NE_a"]  / 100.0) ** 2)
mse_opp = np.nanmean((df["logits_a"] - df["p_BR_to_opp"]  / 100.0) ** 2)

print("Nash mse:", mse_nash)
print("Opponent mse:", mse_opp)