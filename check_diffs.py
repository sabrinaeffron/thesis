import pandas as pd

# see number of rows with same Nash and opponent optimal responses
df = pd.read_csv("outputs_ood/olmo-3-7b/logits_test_results_swap3_1000_0.csv")
filtered = df[df["p_NE_a"] != df["p_BR_to_opp"]]

print(sum(df['p_NE_a'] == 0))
print(sum(df['p_BR_to_opp'] == 0))

# print(sum(df['p_NE_a'] == df['p_BR_to_opp']))
# print(sum(df['p_NE_a'] != df['p_BR_to_opp']))

# print(sum(df['p_NE_a'] == df['p_BR_to_opp']) / len(df))
