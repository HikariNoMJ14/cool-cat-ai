from difflib import SequenceMatcher
import pandas as pd


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


version = '0.7'

df = pd.read_csv(f'data/Thesis - Jazz Dataset - v{version}.csv')

song_name = df['Song name']

for index, row in df.iterrows():
    for index2, row2 in df.iterrows():
        score = similar(row['Song name'].strip(), row2['Song name'].strip())

        if score > 0.75 and score != 1:
           print(row['Source'] + '-' + row['Song name'] + ' ---- ' + row2['Source'] + '-' + row2['Song name'])