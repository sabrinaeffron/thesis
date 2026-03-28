"""
Convert human data to a dataset that can be used for SFT testing
"""

import pandas as pd
from utils import get_sft_prefix

SFT_PREFIX = get_sft_prefix()

df = pd.read_csv('g2p2k/g2p2k_test_swap.csv')
print('number of testing problems:', len(df))

sft_texts = []
for i, row in df.iterrows():
    gamble_text = row['text']
    sft_text = SFT_PREFIX + gamble_text
    sft_texts.append(sft_text)

df['sft_text'] = sft_texts
df.to_csv('g2p2k/g2p2k_test_swap_bare.csv', index=False)
