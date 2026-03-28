import re
import numpy as np
import pandas as pd

CSV_PATH = "g2p2k/games2p2k_text_swap.csv"
df = pd.read_csv(CSV_PATH)

# Parse payoff matrices from text
PAT = re.compile(
    r"row player chooses Option\s+([AB]).*?"
    r"column player chooses Option\s+([CD]).*?"
    r"row player gets\s+(\d+)\s+and\s+the\s+column player gets\s+(\d+)",
    re.IGNORECASE | re.DOTALL
)

def parse_matrices(text: str):
    """
    Returns:
      R: row payoff matrix, shape (2,2) with rows [A,B] and cols [C,D]
      C: col payoff matrix, shape (2,2) with rows [A,B] and cols [C,D]
    """
    R = np.full((2, 2), np.nan, dtype=float)
    C = np.full((2, 2), np.nan, dtype=float)

    matches = PAT.findall(text)
    for r_opt, c_opt, r_pay, c_pay in matches:
        i = 0 if r_opt == "A" else 1
        j = 0 if c_opt == "C" else 1
        R[i, j] = float(r_pay)
        C[i, j] = float(c_pay)

    return R, C

# Payoff-dominant pure Nash equilibrium
# return 1 if equilibrium is option A and 0 if option B
def pure_nash(R, C):
    best_val = -np.inf
    best_ne = None
    for i in (0, 1):
        for j in (0, 1):
            if R[i, j] >= np.max(R[:, j]) and C[i, j] >= np.max(C[i, :]):
                val = R[i, j] + C[i, j]
                if val > best_val:
                    best_ne = (i, j)
                    best_val = val
    if best_ne is None:
        print("nan nash")
        return np.nan
    return 1.0 if best_ne[0] == 0 else 0.0

# Best response to empirical opponent behavior
def best_response_to_q(R, q, tol=1e-12):
    """
    q = P(opponent plays C)
    Returns p_BR in {0, 0.5, 1} for choosing A (hard best response).
    """
    EU_A = q * R[0, 0] + (1 - q) * R[0, 1]
    EU_B = q * R[1, 0] + (1 - q) * R[1, 1]
    if abs(EU_A - EU_B) < tol:
        return 0.5
    return 1.0 if EU_A > EU_B else 0.0

# Build opponent empirical rate q_human using the flipped instance in same game_id
opp_q = pd.Series(index=df.index, dtype=float)
for _, g in df.groupby(["game_id", "version"]):
    i, j = list(g.index)
    opp_q.loc[i] = df.loc[j, "aRate"]
    opp_q.loc[j] = df.loc[i, "aRate"]

df["opp_q"] = opp_q  # P(opponent plays C)

# Compute outputs for every row
p_NE_list = []
p_BR_list = []

for _, row in df.iterrows():
    R, C = parse_matrices(row["text"])

    p_NE_list.append(pure_nash(R, C))
    p_BR_list.append(best_response_to_q(R, float(row["opp_q"])))

df["p_NE_a"] = p_NE_list
df["p_BR_to_opp"] = p_BR_list

# Save
OUT_PATH = "g2p2k/games2p2k_text_swap_responses.csv"
df.to_csv(OUT_PATH, index=False)
print(f"Saved: {OUT_PATH}")
