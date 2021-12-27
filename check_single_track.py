import os
from glob import glob
import music21
import multiprocessing
import pretty_midi

import warnings

warnings.filterwarnings("ignore")

folder = "./data/tmp"

files = [y for x in os.walk(folder) for y in glob(os.path.join(x[0], '*.mid'))]


def check_single_track(file):
    stream = music21.converter.parse(file, forceSource=True)

    print([p.partName for p in stream.parts])

    if len(stream.parts) == 1:
        return True
    else:
        return False

# print(files)

# for f in files:
#     check_single_track(f)

pool = multiprocessing.Pool(50)

pool.map(check_single_track, files)
pool.close()
pool.join()
