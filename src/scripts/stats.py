import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

version = '1.1'

df = pd.read_csv(f'../../data/Thesis - Jazz Dataset/Thesis - Jazz Dataset - v{version}.csv')

df['Song name'] = df['Song name'].str.strip()
# df['Song name'] = df['Song name'].str.lower()
# df['Song name'] = df['Song name'].str.replace(r"[^0-9a-zA-Z]+", "")

orig = df[df['Original'] == 'yes']
impro = df[df['Original'] != 'yes']

print(f"Songs with original melody: {orig.shape[0]}")
print(f"Songs with improvised melody: {impro.shape[0]}")

impro.loc[:, 'Has Original'] = impro['Song name'].isin(orig['Song name'])

has_original = impro[impro['Has Original']]
no_original = impro[~impro['Has Original']]
no_original["multiple"] = no_original[['Song name']].duplicated(keep=False)
no_original = no_original[no_original['multiple']].sort_values('Song name')

orig.loc[:, 'Has Impro'] = orig['Song name'].isin(has_original['Song name'])
has_impro = orig[orig['Has Impro']]

has_original.to_csv(f'../../data/has_original_v{version}.csv')
has_impro.to_csv(f'../../data/has_impro_v{version}.csv')
no_original.to_csv(f'../../data/no_original_v{version}.csv')

# sns.countplot(x=has_original['Song name'].value_counts())
# plt.show()

print(f"Complete examples: {has_original['Song name'].shape[0]}")
print(f"Unique songs: {has_original['Song name'].unique().shape[0]}")
print(f"Songs with more than one improvised version: {has_original['Song name'].value_counts()[has_original['Song name'].value_counts() > 1].shape[0]}")
print(f'Potentially complete examples: {no_original.shape[0]}')

# sns.countplot(x=df['Song name'].value_counts())
# plt.show()

