import pandas as pd

from src.utils import similar

if __name__ == '__main__':
    version = '1.0'

    df = pd.read_csv(f'../data/Thesis - Jazz Dataset/Thesis - Jazz Dataset - v{version}.csv')

    song_name = df['Song name']

    for index, row in df.iterrows():
        for index2, row2 in df.iterrows():
            # print(row2['Song name'])
            score = similar(
                row['Song name'].strip(),
                row2['Song name'].strip()
            )

            if score > 0.75 and score != 1:
               print(row['Source'] + ' - ' + row['Song name'] + ' ---- ' + row2['Source'] + ' - ' + row2['Song name'])
