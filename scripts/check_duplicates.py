import os
from glob import glob
from multiprocessing import Pool

folder = '/media/manu/DATA/Mac/Documents/University/Thesis/Unique Raw Data'

files = [y for x in os.walk(folder) for y in glob(os.path.join(x[0], '*.mid'))]

print(len(files))


def remove_duplicates(s1):
    for s2 in files:
        if s1 == s2:
            return
        if os.path.exists(s1) and os.path.exists(s2) and os.path.getsize(s1) == os.path.getsize(s2):
            with open(s1, 'rb') as fobj1:
                with open(s2, 'rb') as fobj2:
                    if fobj1.read(2048) == fobj2.read(2048):
                        print(s2)
                        os.remove(s2)


pool = Pool(50)  # Create a multiprocessing Pool
pool.map(remove_duplicates, files)
