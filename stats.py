import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/Thesis - Jazz Dataset - v0.1.csv')

df['Song name'] = df['Song name'].str.strip()

orig = df[df['Original'] == 'yes']
impro = df[df['Original'] != 'yes']

print(f"Songs with original melody: {orig.shape[0]}")
print(f"Songs with improvised melody: {impro.shape[0]}")

impro.loc[:, 'Has Original'] = impro['Song name'].isin(orig['Song name'])

has_original = impro[impro['Has Original']]
no_original = impro[~impro['Has Original']]

# sns.countplot(x=has_original['Song name'].value_counts())
# plt.show()

print(f"Complete examples: {has_original['Song name'].shape[0]}")
print(f"Unique songs: {has_original['Song name'].unique().shape[0]}")
print(f"Songs with more than one improvised version: {has_original['Song name'].value_counts()[has_original['Song name'].value_counts() > 1].shape[0]}")

# sns.countplot(x=df['Song name'].value_counts())
# plt.show()

