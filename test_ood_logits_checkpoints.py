import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import torch.nn.functional as F
import numpy as np

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["WANDB_DISABLED"] = "true"
os.environ["DISABLE_MMR_VISION"] = "1"

PROMPT1 = "In this experiment, you will play a simple two-player game. Both you and another player will make a choice, and the payoff each of you receives depends on the combination of both players' choices. You will act as the row player, and your opponent will act as the column player. \n\n"
PROMPT2 = "\n\nAfter viewing the payoff matrix, you must choose between Option A and Option B. Please respond with exactly one of the following options and nothing else:\nOption A\nOption B"

ANS = "Option "

nums = list(range(100, 6500, 400))
nums.insert(0, 0)
nums.append(6522)

# Qwen-7B-SFT
# Gemma-2-9B
# olmo-3-7b
# Llama-3.1-8B
model_dir_name = "Qwen-7B-SFT"
seed = "5000"
device = "cuda"

print(model_dir_name)

if "Qwen" in model_dir_name:
    model_name = "qwen/Qwen-2.5-7B-Instruct"
    output_dir = "/scratch/gpfs/GRIFFITHS/se1854/test_results/qwen-7b-sft/ood_test_results/"
    test_folder = "/scratch/gpfs/GRIFFITHS/se1854/g2p2k/g2p2k_test_swap_qwen.csv"
elif "Gemma" in model_dir_name:
    model_name = "google/Gemma-2-9B-Instruct"
    output_dir = "/scratch/gpfs/GRIFFITHS/se1854/test_results/gemma-2-9b/ood_test_results/"
    test_folder = "/scratch/gpfs/GRIFFITHS/se1854/g2p2k/g2p2k_test_swap_gemma.csv"
elif "olmo" in model_dir_name:
    model_name = "allenai/Olmo-3-7B-Instruct"
    output_dir = "/scratch/gpfs/GRIFFITHS/se1854/test_results/olmo-3-7b/ood_test_results/"
    test_folder = "/scratch/gpfs/GRIFFITHS/se1854/g2p2k/g2p2k_test_swap_olmo.csv"
elif "Llama" in model_dir_name:
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    output_dir = "/scratch/gpfs/GRIFFITHS/se1854/test_results/llama-3.1-8b/ood_test_results/"
    test_folder = "/scratch/gpfs/GRIFFITHS/se1854/g2p2k/g2p2k_test_swap_llama.csv"

def get_choice_logits(prompt, model):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

    # logits for next token prediction
    next_token_logits = outputs.logits[:, -1, :]  # (1, vocab)

    # token IDs for "A" and "B"
    token_0 = tokenizer.encode("A", add_special_tokens=False)[0]
    token_1 = tokenizer.encode("B", add_special_tokens=False)[0]

    logits_0 = next_token_logits[0, token_0].item()
    logits_1 = next_token_logits[0, token_1].item()

    return logits_0, logits_1

for checkpoint in nums:
    if "Qwen" in model_dir_name:
        if checkpoint == 0:
            model_dir = "/scratch/gpfs/GRIFFITHS/se1854/models/Qwen2.5-7B-Instruct"
        else:
            model_dir = "/scratch/gpfs/GRIFFITHS/se1854/outputs/Qwen-7B-SFT_swap"+str(seed)+"/checkpoint-"+str(checkpoint)
    elif "Gemma" in model_dir_name:
        if checkpoint == 0:
            model_dir = "/scratch/gpfs/GRIFFITHS/se1854/models/gemma-2-9b-it"
        else:
            model_dir = "/scratch/gpfs/GRIFFITHS/se1854/outputs/Gemma-2-9B_swap"+str(seed)+"/checkpoint-"+str(checkpoint)
    elif "olmo" in model_dir_name:
        if checkpoint == 0:
            model_dir = "/scratch/gpfs/GRIFFITHS/se1854/models/olmo-3-7b-instruct"
        else:
            model_dir = "/scratch/gpfs/GRIFFITHS/se1854/outputs/olmo-3-7b_swap"+str(seed)+"/checkpoint-"+str(checkpoint)
    elif "Llama" in model_dir_name:
        if checkpoint == 0:
            model_dir = "/scratch/gpfs/GRIFFITHS/se1854/models/Meta-Llama-3.1-8B-Instruct"
        else:
            model_dir = "/scratch/gpfs/GRIFFITHS/se1854/outputs/Llama-3.1-8B_swap"+str(seed)+"/checkpoint-"+str(checkpoint)

    print(model_dir)

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

    df = pd.read_csv(test_folder)
    logits_a, logits_b = [], []
    for _, row in df.iterrows():
        chat = [
                {'role': 'user', 'content': PROMPT1 + row["text"] + PROMPT2},
                {'role': 'assistant', 'content': ANS}
                ]  # create conversation, input user question
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False) 
        logits_zero, logits_one = get_choice_logits(prompt, model)

        # normalize and get probabilities
        logits = torch.tensor([logits_zero, logits_one])
        probs = F.softmax(logits, dim=0)

        p_choice_0 = probs[0].item()
        p_choice_1 = probs[1].item()

        print('='*10)
        print("a: "+str(p_choice_0))
        print("b: "+str(p_choice_1))

        logits_a.append(p_choice_0)
        logits_b.append(p_choice_1)

    # Save results
    df["logits_a"] = logits_a
    df["logits_b"] = logits_b

    # add best responses
    full = pd.read_csv("/scratch/gpfs/GRIFFITHS/se1854/g2p2k/games2p2k_text_swap_responses.csv")
    cols = ["game_id", "text", "p_NE_a", "p_BR_to_opp", "opp_q"]
    df_response = df.merge(full[cols], on=["game_id", "text"], how="left")

    mse_nash = np.nanmean((df_response["logits_a"] - df_response["p_NE_a"]) ** 2)
    mse_opp = np.nanmean((df_response["logits_a"] - df_response["p_BR_to_opp"]) ** 2)

    print("Nash mse:", mse_nash)
    print("Opponent mse:", mse_opp)

    df_response.to_csv(output_dir+"logits_test_results_swap3_"+str(seed)+"_"+str(checkpoint)+".csv", index=False)
    print("Saved predictions")

    del model
    torch.cuda.empty_cache()
