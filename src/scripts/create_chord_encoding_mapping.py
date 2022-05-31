import json

import numpy as np

from src.ezchord import Chord
from src.utils import get_chord_progressions
from src.melody import Melody

if __name__ == "__main__":
    n_extensions = 12
    encoding_type = 'compressed'

    all_chords = {}
    cp = get_chord_progressions()

    for song_name, song_structure in cp.items():
        for section, chords in song_structure['progression'].items():
            for chord_name in chords:
                if chord_name not in all_chords.keys():
                    chord = Chord(chord_name)
                    notes = chord.getMIDI()

                    if encoding_type == 'extended':
                        encoded = list([int(p)
                                        for p in Melody.extended_chord_encoding(notes, n_extensions)])
                    elif encoding_type == 'fixed':
                        encoded = list([int(p)
                                        if not np.isnan(p) else None
                                        for p in Melody.fixed_chord_encoding(notes, n_extensions)])
                    elif encoding_type == 'compressed':
                        encoded = list([int(p)
                                        if not np.isnan(p) else None
                                        for p in Melody.compressed_chord_encoding(notes)])
                    else:
                        raise Exception(f"Chord encoding {encoding_type} doesn't exists")

                    all_chords[chord_name] = encoded

    json.dump(all_chords, open(f'../../data/tensor_dataset/chords/{encoding_type}_{n_extensions}.json', 'w+'))


