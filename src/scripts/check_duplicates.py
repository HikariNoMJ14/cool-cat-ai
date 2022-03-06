import os
from glob import glob
import itertools
from multiprocessing import Pool

# folder = '../../data/Unique Raw Data'
folder = '/media/manu/MUSIC/Mac/Documents/University/Thesis/Unique Raw Data'

files = [y for x in os.walk(folder) for y in glob(os.path.join(x[0], '*.mid'))]

print(len(files))


def remove_duplicate(f1, f2):
    if os.path.exists(f1) and os.path.exists(f2) and os.path.getsize(f1) == os.path.getsize(f2):
        with open(f1, 'rb') as fobj1:
            with open(f2, 'rb') as fobj2:
                if fobj1.read(2048) == fobj2.read(2048):
                    rm = f2
                    if 'JazzStandards' in f1:
                        rm = f1
                    if 'JazzStandards' in f2:
                        rm = f2

                    os.remove(rm)

# pool = Pool(20)  # Create a multiprocessing Pool
# pool.map(remove_duplicates, list(itertools.combinations(files, 2)))


for f1, f2 in list(itertools.combinations(files, 2)):
    remove_duplicate(f1, f2)

