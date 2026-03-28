import pandas as pd

# check if outputs are valid
nums = list(range(200,1601,200))
nums.append(1632)
for n in nums:
    df = pd.read_csv("/home/se1854/Thesis/g2p2k_test_results_"+str(n)+".csv")
    print("checkpoint "+str(n))
    for _, row in df.iterrows():
        if row["pred_A"] + row["pred_B"] != 100:
            print("values do not sum to 100")
            print("values are "+{row["pred_A"]}+" and "+{row["pred_B"]})
    
