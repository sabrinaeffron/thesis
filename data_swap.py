import re
import numpy as np
import pandas as pd

CSV_PATH = "g2p2k/g2p2k_test.csv"
df = pd.read_csv(CSV_PATH)

full_dict = {
    'order': [],
    'game_type': [],
    'game_id': [],
    'unique_id': [],
    'text': [],
    'aRate': [],
    'version': []
}

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
    R = np.full((2, 2), np.nan, dtype=int)
    C = np.full((2, 2), np.nan, dtype=int)

    matches = PAT.findall(text)
    for r_opt, c_opt, r_pay, c_pay in matches:
        i = 0 if r_opt == "A" else 1
        j = 0 if c_opt == "C" else 1
        R[i, j] = r_pay
        C[i, j] = c_pay

    return R, C

def make_text(row, col):
    return (
        f"If the row player chooses Option A and the column player chooses Option C, "
        f"the row player gets {row[0,0]} and the column player gets {col[0,0]};\n"
        f"If the row player chooses Option A and the column player chooses Option D, "
        f"the row player gets {row[0,1]} and the column player gets {col[0,1]};\n"
        f"If the row player chooses Option B and the column player chooses Option C, "
        f"the row player gets {row[1,0]} and the column player gets {col[1,0]};\n"
        f"If the row player chooses Option B and the column player chooses Option D, "
        f"the row player gets {row[1,1]} and the column player gets {col[1,1]}.\n"
    )

for i, row in df.iterrows():
    R, C = parse_matrices(row["text"])
    if np.isnan(R).any():
        print("NaNs in R:", np.argwhere(np.isnan(R)))
    if np.isnan(C).any():
        print("NaNs in C:", np.argwhere(np.isnan(C)))

    even = True
    if i % 2 == 1:
        even = False

    # original
    full_dict['order'].append(row['order'])
    full_dict['game_type'].append(row['game_type'])
    full_dict['game_id'].append(row['game_id'])
    full_dict['unique_id'].append(row['unique_id'])
    full_dict['text'].append(row['text'])
    full_dict['aRate'].append(row['aRate'])
    full_dict['version'].append(0)

    # swap rows
    full_dict['order'].append(row['order'])
    full_dict['game_type'].append(row['game_type'])
    full_dict['game_id'].append(row['game_id'])
    full_dict['unique_id'].append(row['unique_id'])
    R_rows = np.flip(R, axis=0)
    C_rows = np.flip(C, axis=0)
    text_rows = make_text(R_rows, C_rows)
    full_dict['text'].append(text_rows)
    full_dict['aRate'].append(1 - row['aRate'])
    if even:
        full_dict['version'].append(1)
    else:
        full_dict['version'].append(2)

    # swap cols
    full_dict['order'].append(row['order'])
    full_dict['game_type'].append(row['game_type'])
    full_dict['game_id'].append(row['game_id'])
    full_dict['unique_id'].append(row['unique_id'])
    R_cols = np.flip(R, axis=1)
    C_cols = np.flip(C, axis=1)
    text_cols = make_text(R_cols, C_cols)
    full_dict['text'].append(text_cols)
    full_dict['aRate'].append(row['aRate'])
    if even:
        full_dict['version'].append(2)
    else:
        full_dict['version'].append(1)

    # swap rows and cols
    full_dict['order'].append(row['order'])
    full_dict['game_type'].append(row['game_type'])
    full_dict['game_id'].append(row['game_id'])
    full_dict['unique_id'].append(row['unique_id'])
    R_both = np.flip(R, axis=(0, 1))
    C_both = np.flip(C, axis=(0, 1))
    text_both = make_text(R_both, C_both)
    full_dict['text'].append(text_both)
    full_dict['aRate'].append(1 - row['aRate'])
    full_dict['version'].append(3)

full_df = pd.DataFrame(full_dict)

# Save
OUT_PATH = "g2p2k/g2p2k_test_swap.csv"
full_df.to_csv(OUT_PATH, index=False)
print(f"Saved: {OUT_PATH}")
