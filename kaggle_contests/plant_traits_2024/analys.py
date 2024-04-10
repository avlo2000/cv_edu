import pandas as pd

df = pd.read_csv('planttraits2024/train.csv')
for c in df.columns:
    print(c)
print(df.head(5))
print(len(df))
