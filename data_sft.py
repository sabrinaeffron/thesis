"""
Convert human training data to a dataset that can be used for SFT training
"""

import pandas as pd
from utils import get_sft_prefix, get_sft_prefix_sys, get_sft_prefix_assistant
from transformers import AutoTokenizer
model_dir = "/scratch/gpfs/GRIFFITHS/se1854/models/gemma-2-9b-it"
tokenizer = AutoTokenizer.from_pretrained(model_dir, 
                                          local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token

SFT_PREFIX = get_sft_prefix()
SYS_SFT_PREFIX = get_sft_prefix_sys()
ASSISTANT_SFT_PREFIX = get_sft_prefix_assistant()

# Generate SFT text based on RL's training problems (deduplicated)
df = pd.read_csv('/scratch/gpfs/GRIFFITHS/se1854/g2p2k/g2p2k_train_swap.csv')
print('number of training problems:', len(df))

messages, centaur_texts = [], []
for i, row in df.iterrows():
    gamble_text = row['text']
    aRate = row['aRate']
    sft_answer = """\n```json\n""" +\
                    """{\n"""+\
                    f"""  "option_A": {int(aRate*100)},\n"""+\
                    f"""  "option_B": {100-int(aRate*100)}\n"""+\
                    """}\n""" +\
                    """```"""
    message = [
                    # {'role': 'system', 'content': SYS_SFT_PREFIX}, # Gemma does not support system message
                    {
                        'role': 'user', 
                        'content': SFT_PREFIX + gamble_text,
                    },
                    {
                        'role': 'assistant', 
                        'content': sft_answer,
                    }
                ]
    message = tokenizer.apply_chat_template(message, tokenize=False)
    messages.append(message)

df['messages'] = messages
df.to_csv('/scratch/gpfs/GRIFFITHS/se1854/g2p2k/g2p2k_train_swap_gemma.csv', index=False)
