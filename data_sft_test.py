"""
Convert human data to a dataset that can be used for testing
"""

import pandas as pd
from utils import get_sft_prefix, get_sft_prefix_sys, get_sft_prefix_assistant
from transformers import AutoTokenizer

# Qwen2.5-7B-Instruct
# gemma-2-9b-it
# olmo-3-7b-instruct
# Meta-Llama-3.1-8B-Instruct
model_dir = "/scratch/gpfs/GRIFFITHS/se1854/models/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_dir, 
                                          local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token

SFT_PREFIX = get_sft_prefix()
SYS_SFT_PREFIX = get_sft_prefix_sys()
ASSISTANT_SFT_PREFIX = get_sft_prefix_assistant()

df = pd.read_csv('/scratch/gpfs/GRIFFITHS/se1854/g2p2k/g2p2k_test_swap.csv')
print('number of testing problems:', len(df))

sft_texts = []
for i, row in df.iterrows():
    gamble_text = row['text']
    sft_chat = [
        {'role': 'system', 'content': SYS_SFT_PREFIX}, # Gemma does not support system message
        {'role': 'user', 'content': ASSISTANT_SFT_PREFIX + gamble_text},
        ]
    sft_text = tokenizer.apply_chat_template(
        sft_chat, 
        tokenize=False, 
        add_generation_prompt=True
    )
    sft_texts.append(sft_text)

df['sft_text'] = sft_texts
df.to_csv('/scratch/gpfs/GRIFFITHS/se1854/g2p2k/g2p2k_test_swap_llama.csv', index=False)
