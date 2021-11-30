import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

version = '0.8'

df = pd.read_csv(f'data/Thesis - Jazz Dataset - v{version}.csv')

df['Song name'] = df['Song name'].str.strip()
df['Song name 2'] = df['Song name'].str.lower()
df['Song name 2'] = df['Song name 2'].str.replace(r"[^0-9a-zA-Z]+", "")

orig = df[df['Original'] == 'yes']
impro = df[df['Original'] != 'yes']

print(f"Songs with original melody: {orig.shape[0]}")
print(f"Songs with improvised melody: {impro.shape[0]}")

impro.loc[:, 'Has Original'] = impro['Song name 2'].isin(orig['Song name 2'])

has_original = impro[impro['Has Original']]
no_original = impro[~impro['Has Original']]

orig.loc[:, 'Has Impro'] = orig['Song name 2'].isin(has_original['Song name 2'])
has_impro = orig[orig['Has Impro']]

has_original.to_csv(f'data/has_original_v{version}.csv')
has_impro.to_csv(f'data/has_impro_v{version}.csv')

# sns.countplot(x=has_original['Song name'].value_counts())
# plt.show()

print(f"Complete examples: {has_original['Song name 2'].shape[0]}")
print(f"Unique songs: {has_original['Song name 2'].unique().shape[0]}")
print(f"Songs with more than one improvised version: {has_original['Song name 2'].value_counts()[has_original['Song name 2'].value_counts() > 1].shape[0]}")

# sns.countplot(x=df['Song name'].value_counts())
# plt.show()

