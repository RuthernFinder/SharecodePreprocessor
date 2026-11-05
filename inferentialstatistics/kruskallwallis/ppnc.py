import pandas as pd

CSV_FILE = "../../GPU_10_Features_Processed_Final.csv"
GROUPS = ["DDR", "DDR2", "DDR3", "GDDR3", "GDDR5"]
VALUE, GROUP = "Memory_Bandwidth_GBs", "Memory_Type"

df = pd.read_csv(CSV_FILE)
df = df[df[GROUP].isin(GROUPS)].dropna(subset=[VALUE])

# Trung bình, trung vị, phương sai mẫu của từng nhóm
stat = df.groupby(GROUP)[VALUE].agg(
    n="size",
    mean="mean",
    median="median",
    variance="var",          # phương sai mẫu (chia n - 1)
).loc[GROUPS]

print(stat.round(2).to_string())
