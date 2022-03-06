from shutil import copy
import os
import pandas as pd

version = '1.2'


def copy_files(data, src_folder, dst_folder):
    for idx, row in data.iterrows():
        src_filepath = os.path.join(src_folder, f'{row["source"]}/{row["filename"]}')
        dst_filepath = os.path.join(dst_folder, f'{row["source"]}/{row["filename"]}')

        try:
            if not os.path.exists(os.path.dirname(dst_filepath)):
                os.makedirs(os.path.dirname(dst_filepath))
            copy(src_filepath, dst_filepath)
        except OSError as e:
            print(e, src_filepath)


if __name__ == "__main__":
    has_original = pd.read_csv(f'../../data/intermediate_csvs/v{version}/has_original_v{version}.csv')

    src_filepath = f'../../data/Unique Raw Data/v{version}'
    dst_filepath = f'../../data/Complete Examples/v{version}'

    copy_files(has_original, src_filepath, dst_filepath)

    no_impro = pd.read_csv(f'../../data/intermediate_csvs/v{version}/no_original_v{version}.csv')

    src_filepath = f'../../data/Unique Raw Data/v{version}'
    dst_filepath = f'../../data/Complete Examples Maybe/v{version}'

    copy_files(has_original, src_filepath, dst_filepath)

