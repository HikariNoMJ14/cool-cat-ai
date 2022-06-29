import os
import sys
import difflib
import shutil
from glob import glob
import pandas as pd
import numpy as np
import operator

import music21

sys.path.append('../..')
from src.utils import get_chord_progressions, calculate_melody_results, flatten_chord_progression
from src.melody import Melody

source = 'Jazz-Midi'

input_folder = f'../data/Complete Examples Melodies Auto/v1.2/{source}'
input_folder2 = f'../data/Complete Examples Melodies Random/v1.2/{source}'

filepaths = [y for x in os.walk(input_folder) for y in glob(os.path.join(x[0], '*.mid'))]
filepaths = filepaths + [y for x in os.walk(input_folder2) for y in glob(os.path.join(x[0], '*.mid'))]
cps = get_chord_progressions('..')