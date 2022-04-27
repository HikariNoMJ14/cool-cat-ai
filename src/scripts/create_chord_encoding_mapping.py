import json

from ezchord import Chord

from utils import get_chord_progressions
from melody import TimeStepMelody

if __name__ == "__main__":
    all_chords = {}
    cp = get_chord_progressions()

    for song_name, song_structure in cp.items():
        for section, chords in song_structure['progression'].items():
            for chord_name in chords:
                if chord_name not in all_chords.keys():
                    chord = Chord(chord_name)
                    notes = chord.getMIDI()
                    # TODO this is extended-7, generalize
                    encoded = list([int(p) for p in TimeStepMelody.extended_chord_encoding(notes, 7)])

                    all_chords[chord_name] = encoded

    # for k, v in all_chords.items():
    #     print(k , v)

    json.dump(all_chords, open('../../data/tensor_dataset/chords/extended_7.json', 'w+'))


