import os
from glob import glob
import music21
import multiprocessing

folder = "./data/output/Unique Melodies"

files = [y for x in os.walk(folder) for y in glob(os.path.join(x[0], '*.mid'))]


def check_single_track(file):
    stream = music21.converter.parse(file, forceSource=True)

    if len(stream.parts) == 1:
        return True
    else:
        print("Error", [p.partName for p in stream.parts])
        return False


for f in files:
    check_single_track(f)

# pool = multiprocessing.Pool(50)
#
# pool.map(filter_melody, files)
# pool.close()
# pool.join()
