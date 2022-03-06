import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from glob import glob

version = '1.2'
folder = '../../data/Unique Raw Data'

files = [y for x in os.walk(folder) for y in glob(os.path.join(x[0], '*.mid'))]

data = [(l.split('/')[-2], l.split('/')[-1]) for l in files]

df = pd.DataFrame().from_dict(data)
df.columns = ['source', 'filename']

df.loc[:, 'original'] = df['source'] == 'Real Book'

df.loc[:, 'songname'] = df['filename'].str.replace('.mid', '').str.replace(r' \([0-9]\)', '', regex=True).str.replace(r'.* - (.*)', r'\1', regex=True)
df.loc[:, 'songname_low'] = df['songname'].str.lower()

# print(df)

orig = df[df['original']]
impro = df[~df['original']]

all_orig = set(orig['songname_low'].values)

print(f"Songs with original melody: {orig.shape[0]}")
print(f"Songs with improvised melody: {impro.shape[0]}")

impro.loc[:, 'has_original'] = impro['songname_low'].isin(all_orig)

has_original = impro[impro['has_original']]
no_original = impro[~impro['has_original']]
no_original.loc[:, "multiple"] = no_original[['songname_low']].duplicated(keep=False)
no_original = no_original[no_original['multiple']].sort_values('songname')

all_impro = set(has_original['songname_low'].values)
orig.loc[:, 'has_impro'] = orig['songname_low'].isin(all_impro)
has_impro = orig[orig['has_impro']]

unique_renamed_filename = f'../../data/intermediate_csvs/v{version}/unique_renamed_v{version}.csv'
if not os.path.exists(unique_renamed_filename):
    has_original.to_csv(unique_renamed_filename)
else:
    print('File already exists!')

has_original_filename = f'../../data/intermediate_csvs/v{version}/has_original_v{version}.csv'
if not os.path.exists(has_original_filename):
    has_original.to_csv(has_original_filename)
else:
    print('File already exists!')

has_impro_filename = f'../../data/intermediate_csvs/v{version}/has_impro_v{version}.csv'
if not os.path.exists(has_impro_filename):
    has_impro.to_csv(has_impro_filename)
else:
    print('File already exists!')

no_original_filename = f'../../data/intermediate_csvs/v{version}/no_original_v{version}.csv'
if not os.path.exists(no_original_filename):
    no_original.to_csv(no_original_filename)
else:
    print('File already exists!')

# sns.countplot(x=no_original['songname'].value_counts())
# plt.show()

print(f"Complete examples: {has_original['songname'].shape[0]}")
print(f"Unique songs: {has_original['songname'].unique().shape[0]}")
print(f"Songs with more than one improvised version: {has_original['songname'].value_counts()[has_original['songname'].value_counts() > 1].shape[0]}")
print(f'Potentially complete examples: {no_original.shape[0]}')

# print(df['songname'].value_counts())
# sns.countplot(x=df['songname'].value_counts())
# plt.show()
