import multiprocessing
import pandas as pd
import music21
from functools import partial
import math
import os
from glob import glob


import warnings

warnings.filterwarnings("ignore")

def filter_melody(file):
    # print(file)

    try:
        stream = music21.converter.parse(file, forceSource=True)

        parts = stream.parts

        melody_track = None

        melody_tracks = [p for p in parts if p is not None
                         and p.partName is not None
                         and any(sub in p.partName.lower() for sub in
                            ['solo', 'melody', 'lead'])]

        # if len(melody_tracks) == 0:
        #     parts = [p for p in parts if p is not None
        #              and p.partName is not None
        #              and all(sub not in p.partName.lower() for sub in
        #                     ['bass', 'bajo', 'basso', 'baixo', 'drum', 'percussion', 'bateria', 'chord', 'rhythm',
        #                      'cymbal', 'clap', 'kick', 'snare', 'hh ', 'hats', 'ride', 'kit'])]
        #
        #     print([p.partName for p in parts])
        #     print(len(parts))
        #
        #     if len(parts) == 1:
        #         melody_track = parts[0]
            # else:
                # print('parts:', parts)
                # bad.write(file)

        if len(melody_tracks) == 1:
            melody_track = melody_tracks[0]
        # else:
            # print('duplicate!!!!', [p.partName for p in melody_tracks])
            # bad.write(file)

        if melody_track:
            # print(melody_track.partName, len(melody_track.flat.notesAndRests))
            score = music21.stream.Score()
            score.insert(0, melody_track)

            score.write('midi', fp=file.replace('Complete Examples', 'Complete Examples Melodies'))

        # print('---------')
    except Exception as e:
        print('Can\'t parse midi into stream')
        print(file)
        print(e)


def find_interval(score):
    try:
        key = score.analyze('key')
    except:
        print('NO KEY')
        return None

    print(key)

    if key.mode == 'minor':
        key = key.getRelativeMajor()

    tonic = music21.note.Note(key.tonic.name)
    interval = music21.interval.Interval(music21.note.Note('C'), tonic)

    return interval


folder = './data/Complete Examples'
files = [y for x in os.walk(folder) for y in glob(os.path.join(x[0], '*.mid'))]

transpose = False

good_melodies = []
boh_melodies = []

# for f in files:
#     filter_melody(f)

pool = multiprocessing.Pool(50)

pool.map(filter_melody, files)
pool.close()
pool.join()
