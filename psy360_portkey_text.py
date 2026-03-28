from portkey_ai import Portkey
import os
from dotenv import load_dotenv
import pandas as pd
import re
import json
import numpy as np
import time
import random
import requests
import math
import signal

class APITimeout(Exception):
    pass

def _alarm_handler(signum, frame):
    raise APITimeout("API call timed out")

signal.signal(signal.SIGALRM, _alarm_handler)

session = requests.Session()
session.keep_alive = True

# Load environment variables from the .env file
load_dotenv()

# Import API key from OS environment variables
AI_SANDBOX_KEY = os.getenv("AI_SANDBOX_KEY")

client = Portkey(api_key=AI_SANDBOX_KEY)

print("AI_SANDBOX_KEY present:", bool(AI_SANDBOX_KEY), flush=True)
if not AI_SANDBOX_KEY:
    raise RuntimeError("AI_SANDBOX_KEY is not set")

test_file = '/scratch/gpfs/GRIFFITHS/se1854/g2p2k/g2p2k_test_swap_bare.csv'
output_dir = '/scratch/gpfs/GRIFFITHS/se1854/test_results/psy360/'

# Set the model deployment name that the prompt should be sent to
available_models = [
                    "gpt-5",
                    "o3-mini",
                    "gpt-4o-mini",
                    "gpt-4o", 
                    "gpt-4-turbo",
                    "gpt-35-turbo-16k", 
                    "Llama-3.3-70B-Instruct", 
                    "mistral-small-2503"
                ]

def binary_cross_entropy(y, p):
    if p <= 0 or p >= 1:
        eps=1e-12
        p = min(max(p, eps), 1 - eps)
        return -(y * math.log(p) + (1 - y) * math.log(1 - p))
    else:
        return -(y * math.log(p) + (1 - y) * math.log(1 - p))

# This function will submit a simple text prompt to the chosen model
def text_prompt_example(model_to_be_used, sft_text):
    max_retries = 8
    for attempt in range(max_retries):
        # Establish a connection to your Azure OpenAI instance
        try:
            # print(f"[{model_to_be_used}] calling API (attempt {attempt+1})", flush=True)
            signal.alarm(90)  # seconds
            response = client.chat.completions.create(
                model=model_to_be_used, 
                messages=[{'role': 'user', 'content': sft_text}]
            )
            out = response.choices[0].message.content
            return out

        except Exception as e:
            print(e.message)
            wait = (2 ** attempt) + random.random()
            print(f"Request failed (attempt {attempt+1}/{max_retries}). "
                f"Retrying in {wait:.2f}s... Error:", e)
            time.sleep(wait)
        finally:
            signal.alarm(0)
    return None

# Execute the example functions
if __name__ == "__main__":

    # Test text prompts with all available models
    for model in available_models:
        df = pd.read_csv(test_file)
        # Execute the text prompt example
        print("\nModel: " + model)
        outputs, pred_A, pred_B, losses, sq_err = [], [], [], [], []
        for _, row in df.iterrows():
            out = text_prompt_example(model, row['sft_text'])
            print(out)
            outputs.append(out)
            try: 
                match = re.search(r"\{.*?\}", out, re.DOTALL)
                data = json.loads(match.group(0))
                a = data.get("option_A", 0)
                b = data.get("option_B", 0)
                if a + b == 100:
                    # calculate binary cross entropy and squared error
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
            except:
                print("Incorrect output format")
                a = np.nan
                b = np.nan
                ce = np.nan
                se = np.nan
            
            pred_A.append(a)
            pred_B.append(b)
            losses.append(ce)
            sq_err.append(se)

        df["pred_A"] = pred_A
        df["pred_B"] = pred_B
        df["bce"] = losses
        df["model_output"] = outputs
        df["sq_err"] = sq_err
        df.to_csv(output_dir+"g2p2k_test_results_swap_"+model+".csv", index=False)
        print("Number of bad rows: ", str(df["pred_A"].isnull().sum()), "/", str(len(df)))
        avg_bce = np.nanmean(losses)
        print(f"Average Binary Cross Entropy Loss: {avg_bce:.4f}")
        mse = np.nanmean(sq_err)
        print(f"Mean Squared Error: {mse:.4f}")

        print("Saved predictions to g2p2k_test_results_swap_"+model+".csv")
