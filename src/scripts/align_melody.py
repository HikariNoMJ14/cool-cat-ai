import os
import json
from glob import glob
from datetime import datetime

from melody import Melody
from utils import get_chord_progressions


if __name__ == "__main__":
    # folder = '../data/Complete Examples Melodies/Doug McKenzie'
    folder = '../data/Complete Examples Melodies'

    filepaths = [y for x in os.walk(folder) for y in glob(os.path.join(x[0], '*.mid'))]

    # filepaths = [
    #     '../data/Complete Examples Melodies/Jazz-Midi/All Of Me.mid',
        # '../data/Complete Examples Melodies/Jazz-Midi/Watermelon Man.mid',
        # '../data/Complete Examples Melodies/Real Book/All Of Me.mid',
        # '../data/Complete Examples Melodies/Real Book/All Blues.mid',
        # '../data/Complete Examples Melodies/Oocities/Come Rain Or Come Shine.mid'
        # '../data/Complete Examples Melodies/Real Book/A Felicidade.mid'
        # '../data/Complete Examples Melodies/Real Book/All Of Me.mid'
        # '../data/Complete Examples Melodies/Real Book/Afro Blue.mid'
        # '../data/Complete Examples Melodies/Real Book/Autumn Leaves.mid'
        # '../data/Complete Examples Melodies/Real Book/Ornithology.mid'
        # '../data/Complete Examples Melodies/Real Book/Margie.mid',
        # '../data/Complete Examples Melodies/Real Book/Oleo.mid',
        # '../data/Complete Examples Melodies/Real Book/Blue Train.mid',
        # '../data/Complete Examples Melodies/Real Book/In Your Own Sweet Way.mid',
        # '../data/Complete Examples Melodies/Real Book/Alice In Wonderland.mid',
        # '../data/Complete Examples Melodies/Weimar DB/Bix Beiderbecke - Margie.mid',
        # '../data/Complete Examples Melodies/Weimar DB/Red Garland - Oleo.mid',
        # '../data/Complete Examples Melodies/Weimar DB/Miles Davis - Oleo (1).mid',
        # '../data/Complete Examples Melodies/Weimar DB/Miles Davis - Oleo (2).mid',
        # '../data/Complete Examples Melodies/Weimar DB/John Coltrane - Oleo.mid',
        # '../data/Complete Examples Melodies/Weimar DB/Lee Morgan - Blue Train.mid',
        # '../data/Complete Examples Melodies/Weimar DB/Curtis Fuller - Blue Train.mid',
        # '../data/Complete Examples Melodies/Weimar DB/John Coltrane - Blue Train.mid'
    # ]

    song_scores = {}
    errors = {}
    alignment = {}

    cp = get_chord_progressions()

    for fp in filepaths:
        melody = Melody(fp)
        key = os.path.join(melody.source, melody.filename)

        melody.setup()

        print(melody.source, melody.song_name)

        if melody.song_name not in cp:
            melody.errors.append('No chords!')
        else:
            melody.set_song_structure(cp[melody.song_name])

        if melody.time_signature is None:
            melody.errors.append('no time signature')
        elif melody.time_signature[0] != 4:
            melody.errors.append('time signature not 4/4')

        melody.align_melody()

        if len(melody.errors) == 0:
            alignment[key] = melody.chord_progression_comparison()

        melody.split_melody()

        # song_scores[key] = {
        #     'melody_key': (melody.key.tonic.name,
        #                    melody.key.mode),
        #     'chord_progression_key': (melody.chord_progression_key,
        #                               'minor' if melody.chord_progression_minor else 'major'),
        #     'starting_measure': melody.starting_measure,
        #     'best_score': melody.alignment_best_score,
        #     'all_scores': melody.alignment_all_scores
        # }

        if len(melody.errors) > 0:
            errors[key] = melody.errors
            print(melody.errors)

        del melody

    # json.dump(song_scores,
    #           open(f'../data/alignment_scores/song_scores-{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}.json', 'w+'))
    json.dump(errors,
              open(f'../data/alignment_scores/errors-{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}.json', 'w+'))
    json.dump(alignment,
              open(f'../data/alignment_scores/alignment-{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}.json', 'w+'))

