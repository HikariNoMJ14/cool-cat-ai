from shutil import copy
import os
import pandas as pd

version = '0.8'

has_original = pd.read_csv(f'data/has_original_v{version}.csv')
has_impro = pd.read_csv(f'data/has_impro_v{version}.csv')

for idx, row in has_impro.iterrows():

    src_filepath = f'/media/manu/DATA/Mac/Documents/University/Thesis/Unique Raw Data/{row["Source"]}/{row["Filename"]}.mid'
    dst_filepath = f'data/Complete Examples/{row["Source"]}/{row["Filename"]}.mid'

    try:
        if not os.path.exists(os.path.dirname(dst_filepath)):
            os.mkdir(os.path.dirname(dst_filepath))
        copy(src_filepath, dst_filepath)
    except OSError as e:
        print(e, src_filepath)

    # print(dst_filepath)
