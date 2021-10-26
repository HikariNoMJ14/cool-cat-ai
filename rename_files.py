import os
import pandas as pd


path = "/Users/manu/Desktop/Renamed Raw Data/Jazz Real & New Real Book + Xtras/Midi New Real Book II"

df = pd.read_csv(os.path.join(path, "A lista.csv"), delimiter='\t', names=['name', 'filename'])

for idx, row in df.iterrows():
    old_filename = os.path.join(path, f"{row['filename']}.MID")
    new_filename = f"{row['name'].lower().title()}.mid"

    if os.path.exists(old_filename):
        os.rename(old_filename, os.path.join(path, new_filename))
