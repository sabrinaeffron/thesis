import json
import pandas as pd
import matplotlib.pyplot as plt

state_path = "outputs/llama-3.1-8b/trainer_state_1000.json"

with open(state_path, "r") as f:
    state = json.load(f)

logs = pd.DataFrame(state["log_history"])

train = logs[logs["loss"].notna()][["step", "loss"]].dropna().sort_values("step")

plt.figure()
plt.plot(train["step"], train["loss"], color="blue", linewidth=2)
plt.xlabel("Training step")
plt.ylabel("Training loss")
plt.title("Training learning curve for Llama-3.1-8B")
plt.tight_layout()
plt.show()

final_step = int(train["step"].iloc[-1])
print("Final step:", final_step)