import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import json
import re
import numpy as np
import math

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["WANDB_DISABLED"] = "true"
os.environ["DISABLE_MMR_VISION"] = "1"

# load model
model_dir = "/scratch/gpfs/GRIFFITHS/se1854/models/gemma-2-9b-it"
# Meta-Llama-3.1-8B-Instruct
# olmo-3-7b-instruct
# Qwen2.5-7B-Instruct
# gemma-2-9b-it

if "Qwen" in model_dir:
    model_name = "qwen/Qwen-2.5-7B-Instruct"
    output_dir = "/scratch/gpfs/GRIFFITHS/se1854/test_results/qwen-7b-sft/"
    test_folder = "/scratch/gpfs/GRIFFITHS/se1854/g2p2k/g2p2k_test_swap_qwen.csv"
elif "gemma" in model_dir:
    model_name = "google/Gemma-2-9B-Instruct"
    output_dir = "/scratch/gpfs/GRIFFITHS/se1854/test_results/gemma-2-9b/"
    test_folder = "/scratch/gpfs/GRIFFITHS/se1854/g2p2k/g2p2k_test_swap_gemma.csv"
elif "olmo" in model_dir:
    model_name = "allenai/Olmo-3-7B-Instruct"
    output_dir = "/scratch/gpfs/GRIFFITHS/se1854/test_results/olmo-3-7b/"
    test_folder = "/scratch/gpfs/GRIFFITHS/se1854/g2p2k/g2p2k_test_swap_olmo.csv"
elif "Llama" in model_dir:
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    output_dir = "/scratch/gpfs/GRIFFITHS/se1854/test_results/llama-3.1-8b/"
    test_folder = "/scratch/gpfs/GRIFFITHS/se1854/g2p2k/g2p2k_test_swap_llama.csv"

device = "cuda"

model = AutoModelForCausalLM.from_pretrained(
    model_dir, 
    dtype=torch.bfloat16,
    attn_implementation="eager",
    device_map=None,
    local_files_only=True,
).to(device)
print(model)
tokenizer = AutoTokenizer.from_pretrained(model_dir, 
                                          local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token

def binary_cross_entropy(y, p):
    if p <= 0 or p >= 1:
        eps=1e-12
        p = min(max(p, eps), 1 - eps)
        return -(y * math.log(p) + (1 - y) * math.log(1 - p))
    else:
        return -(y * math.log(p) + (1 - y) * math.log(1 - p))

def model_completion(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    output_ids = model.generate(input_ids, 
                                max_new_tokens=1024,
                                do_sample=True,
                                temperature=0.7,
                                attention_mask=attention_mask,
                                top_k=70,  # Only consider the top 50 most likely words
                                top_p=0.99,  # Consider tokens with cumulative probability of 90%
                                )
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text

df = pd.read_csv(test_folder)
outputs, pred_A, pred_B, bce, sq_err = [], [], [], [], []
for _, row in df.iterrows():

    output_text = model_completion(row["sft_text"])
    print('='*10)
    print(output_text)
    outputs.append(output_text)

    # extract json from generated text, convert text to real number from json
    try: 
        match = re.search(r"\{.*?\}", output_text, re.DOTALL)
        data = json.loads(match.group(0))
        a = data.get("option_A", 0)
        b = data.get("option_B", 0)
        if a + b == 100:
            # calculate binary cross entropy
            y_true = row["aRate"]
            decimal_a = a / 100.0
            ce = binary_cross_entropy(y_true, decimal_a)
            se = (y_true - decimal_a) ** 2
        else:
            print("Numbers do not sum to 100")
            a = np.nan
            b = np.nan
            ce = np.nan
            se = np.nan
    except Exception as e:
        print("Incorrect output format")
        a = np.nan
        b = np.nan
        ce = np.nan
        se = np.nan

    pred_A.append(a)
    pred_B.append(b)
    bce.append(ce)
    sq_err.append(se)

# Save results
df["pred_A"] = pred_A
df["pred_B"] = pred_B
df["model_output"] = outputs
df["bce"] = bce
df["sq_err"] = sq_err
df.to_csv(output_dir+"test_results_gemma_swap_0.csv", index=False)
print("Saved predictions")
print("Number of bad rows: ", str(df["pred_A"].isnull().sum()), "/", str(len(df)))

avg_bce = np.nanmean(bce)
mse = np.nanmean(sq_err)
print(f"Average Binary Cross Entropy Loss: {avg_bce:.4f}")
print(f"Average Mean Squared Error: {mse:.4f}")
