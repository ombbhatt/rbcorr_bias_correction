import pandas as pd

df = pd.read_csv('comps_yn_rand_2prop_all.csv')
df.sample(n=2100, random_state=42).sort_index().to_csv('comps_yn_rand_2prop_2100.csv', index=False)