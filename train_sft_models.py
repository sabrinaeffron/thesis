import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["WANDB_DISABLED"] = "true"
os.environ["DISABLE_MMR_VISION"] = "1"

# Qwen2.5-7B-Instruct
# gemma-2-9b-it
# olmo-3-7b-instruct
# Meta-Llama-3.1-8B-Instruct
model_dir = "/scratch/gpfs/GRIFFITHS/se1854/models/Meta-Llama-3.1-8B-Instruct" 
seed = 5000

if "Qwen" in model_dir:
    model_name = "qwen/Qwen-2.5-7B-Instruct"
    output_dir = "/scratch/gpfs/GRIFFITHS/se1854/outputs/Qwen-7B-SFT_swap"+str(seed)
    train_file = "/scratch/gpfs/GRIFFITHS/se1854/g2p2k/g2p2k_train_swap_qwen.csv"
elif "gemma" in model_dir:
    model_name = "google/Gemma-2-9B-Instruct"
    output_dir = "/scratch/gpfs/GRIFFITHS/se1854/outputs/Gemma-2-9B_swap"+str(seed)
    train_file = "/scratch/gpfs/GRIFFITHS/se1854/g2p2k/g2p2k_train_swap_gemma.csv"
elif "olmo" in model_dir:
    model_name = "allenai/Olmo-3-7B-Instruct"
    output_dir = "/scratch/gpfs/GRIFFITHS/se1854/outputs/olmo-3-7b_swap"+str(seed)
    train_file = "/scratch/gpfs/GRIFFITHS/se1854/g2p2k/g2p2k_train_swap_olmo.csv"
elif "Llama" in model_dir:
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    output_dir = "/scratch/gpfs/GRIFFITHS/se1854/outputs/Llama-3.1-8B_swap"+str(seed)
    train_file = "/scratch/gpfs/GRIFFITHS/se1854/g2p2k/g2p2k_train_swap_llama.csv"

print('-'*20,f'OUTPUT_DIR={output_dir}')

model = AutoModelForCausalLM.from_pretrained(
    model_dir, 
    dtype=torch.bfloat16,
    attn_implementation="eager", # For Gemma, eager attention is recommended over flash_attention
    # attn_implementation="flash_attention_2",
    device_map=None,
    local_files_only=True,
).to("cuda")
print(model)
tokenizer = AutoTokenizer.from_pretrained(model_dir, 
                                          local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token

# Prepare training data
train_dataset = load_dataset("csv", data_files={"train": train_file})["train"]
train_dataset = train_dataset.shuffle(seed=seed) # shuffle the training set
print(train_dataset.column_names)

# train data
def tokenize_func(batch):
    return tokenizer(batch['messages'], truncation=True, padding=False)
tokenized_train_dataset = train_dataset.map(tokenize_func,
                                            batched=False)
tokenized_train_dataset = tokenized_train_dataset.remove_columns(
    [col for col in tokenized_train_dataset.column_names if col not in ["input_ids", "attention_mask"]]
)

# SFT details
sft_training_args = SFTConfig(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=6, # can change this (og 6, tried 1)
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8, # can change this
    learning_rate=2e-6, # can change this (og 1e-5, tried 2e-6)
    save_steps=100,
    bf16=True,
    gradient_checkpointing=False,
    remove_unused_columns=True,
    report_to="none"
)

peft_config = LoraConfig(
    r=8, # (og 32, tried 8, 16)
    lora_alpha=16, # (og 32, tried 16, try alpha = 2r)
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    task_type="CAUSAL_LM",
    lora_dropout=0.1, # (og 0.05, tried 0.1, 0.2)
)

sft_trainer = SFTTrainer(
    model=model,
    args=sft_training_args,
    train_dataset=tokenized_train_dataset,
    peft_config=peft_config,
)

torch.cuda.empty_cache()
sft_trainer.train()
sft_trainer.model.save_pretrained("fine_tuned_model")
