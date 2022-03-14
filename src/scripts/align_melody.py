import os
import json
from glob import glob
from datetime import datetime

import music21.key
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from melody import Melody
from utils import get_chord_progressions
from objective_metrics import calculate_HC

if __name__ == "__main__":
    version = '1.2'
    quantized = True

    # folder = 'f'../data/Complete Examples Melodies Auto/v{version}/Doug McKenzie'
    folder = f'../../data/Complete Examples Melodies Auto/v{version}/Real Book'

    filepaths = [y for x in os.walk(folder) for y in glob(os.path.join(x[0], '*.mid'))]

    # filepaths = [
        # 'f'../data/Complete Examples Melodies Auto/Jazz-Midi/All Of Me.mid',
        # 'f'../data/Complete Examples Melodies Auto/Jazz-Midi/Watermelon Man.mid',
        # f'../../data/Complete Examples Melodies Auto/v{version}/Real Book/All Of Me.mid',
        # 'f'../data/Complete Examples Melodies Auto/v{version}/Real Book/All Blues.mid',
        # 'f'../data/Complete Examples Melodies Auto/Oocities/Come Rain Or Come Shine.mid'
        # f'../../data/Complete Examples Melodies Auto/v{version}/Real Book/A Felicidade.mid'
        # f'../../data/Complete Examples Melodies Auto/v{version}/Real Book/Ruby My Dear.mid'
        # 'f'../data/Complete Examples Melodies Auto/v{version}/Real Book/All Of Me.mid'
        # 'f'../data/Complete Examples Melodies Auto/v{version}/Real Book/Afro Blue.mid'
        # 'f'../data/Complete Examples Melodies Auto/v{version}/Real Book/Autumn Leaves.mid'
        # f'../../data/Complete Examples Melodies Auto/v{version}/Real Book/New York, New York.mid'
        # 'f'../data/Complete Examples Melodies Auto/v{version}/Real Book/Margie.mid',
        # f'../../data/Complete Examples Melodies Auto/v{version}/Real Book/Ana Maria.mid',
        # f'../../data/Complete Examples Melodies Auto/v{version}/Real Book/I Don\'t Know Why.mid',
        # f'../../data/Complete Examples Melodies Auto/v{version}/Real Book/I Got Rhythm.mid',
        # f'../../data/Complete Examples Melodies Auto/v{version}/Real Book/Black Orpheus.mid',
        # f'../../data/Complete Examples Melodies Auto/v{version}/Real Book/Blue Train.mid',
        # f'../../data/Complete Examples Melodies Auto/v{version}/Real Book/Daahoud.mid',
        # f'../../data/Complete Examples Melodies Auto/v{version}/Real Book/In Your Own Sweet Way.mid',
        # f'../../data/Complete Examples Melodies Auto/v{version}/Real Book/Lazy Bird.mid',
        # f'../../data/Complete Examples Melodies Auto/v{version}/Real Book/Recado Bossa Nova.mid',
        # f'../../data/Complete Examples Melodies Auto/v{version}/Real Book/Giant Steps.mid',
        # f'../../data/Complete Examples Melodies Auto/v{version}/Real Book/Dolores.mid',
        # f'../../data/Complete Examples Melodies Auto/v{version}/Real Book/Epistrophy.mid',
        # f'../../data/Complete Examples Melodies Auto/v{version}/Real Book/Speak No Evil.mid',
        # f'../../data/Complete Examples Melodies Auto/v{version}/Real Book/Monk\'s Mood.mid',
        # f'../../data/Complete Examples Melodies Auto/v{version}/Real Book/Beautiful Friendship.mid',
        # f'../../data/Complete Examples Melodies Auto/v{version}/Real Book/House Of Jade.mid',
        # f'../../data/Complete Examples Melodies Auto/v{version}/Real Book/Equinox.mid',
        # f'../../data/Complete Examples Melodies Auto/v{version}/Real Book/Ruby My Dear.mid',
        # f'../../data/Complete Examples Melodies Auto/v{version}/Real Book/Margie.mid',
        # f'../../data/Complete Examples Melodies Auto/v{version}/Real Book/Esp.mid',
        # f'../../data/Complete Examples Melodies Auto/v{version}/Real Book/Laura.mid',
        # f'../../data/Complete Examples Melodies Auto/v{version}/Real Book/In A Mellow Tone.mid',
        # f'../../data/Complete Examples Melodies Auto/v{version}/Real Book/Come Sunday.mid',
        # 'f'../data/Complete Examples Melodies Auto/v{version}/Real Book/Alice In Wonderland.mid',
        # 'f'../data/Complete Examples Melodies Auto/Weimar DB/Bix Beiderbecke - Margie.mid',
        # 'f'../data/Complete Examples Melodies Auto/Weimar DB/Red Garland - Oleo.mid',
        # 'f'../data/Complete Examples Melodies Auto/Weimar DB/Miles Davis - Oleo (1).mid',
        # 'f'../data/Complete Examples Melodies Auto/Weimar DB/Miles Davis - Oleo (2).mid',
        # 'f'../data/Complete Examples Melodies Auto/Weimar DB/John Coltrane - Oleo.mid',
        # 'f'../data/Complete Examples Melodies Auto/Weimar DB/Lee Morgan - Blue Train.mid',
        # 'f'../data/Complete Examples Melodies Auto/Weimar DB/Curtis Fuller - Blue Train.mid',
        # 'f'../data/Complete Examples Melodies Auto/Weimar DB/John Coltrane - Blue Train.mid'
    # ]

    all_results = []
    errors = {}

    cp = get_chord_progressions()

    starting_measure_exceptions = {
        'Real Book - Beauty And The Beast': 5
    }

    key_exceptions = {
        'Real Book - Dolores': ('Db', 'major'),
        'Real Book - Epistrophy': ('Db', 'major'),
        'Real Book - Giant Steps': ('B', 'major'),
        'Real Book - Speak No Evil': ('C', 'minor'),
        'Real Book - Monk\'s Mood': ('Db', 'major'),
        'Real Book - Beautiful Friendship': ('Db', 'major'),
        'Real Book - House Of Jade': ('C', 'minor'),
        'Real Book - Laura': ('C', 'major'),
        'Real Book - In A Mellow Tone': ('F', 'major'),
        'Real Book - Come Sunday': ('Bb', 'major'),
    }

    for fp in filepaths:
        melody = Melody(fp, version)
        dict_key = os.path.join(melody.source, melody.filename)

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

        starting_measure = 2

        if f'{melody.source} - {melody.song_name}' in starting_measure_exceptions:
            starting_measure = starting_measure_exceptions[f'{melody.source} - {melody.song_name}']
            print(f'Using {starting_measure} for {melody.source} - {melody.song_name}')

        if f'{melody.source} - {melody.song_name}' in key_exceptions:
            tonic = key_exceptions[f'{melody.source} - {melody.song_name}'][0]
            mode = key_exceptions[f'{melody.source} - {melody.song_name}'][1]

            melody.key = music21.key.Key(tonic, mode)
            print(f'Using {melody.key.tonic} for {melody.source} - {melody.song_name}')

        melody.manually_align(starting_measure, quantized)

        if len(melody.errors) > 0:
            errors[dict_key] = melody.errors
        else:
            results = melody.chord_progression_comparison()
            results.update({
                'source': melody.source,
                'filename': melody.filename,
                'starting_measure': melody.starting_measure,
                'melody_mido_key': melody.mido_key,
                # 'melody_music_21_key': melody.music21_key,
                # 'melody_music_21_key2': melody.music21_key2,
                'chord_progression_key': melody.chord_progression_key + 'm'
                if melody.chord_progression_minor
                else melody.chord_progression_key,
                'transpose_semitones': melody.transpose_semitones
            })

            if len(melody.split_note_info) > 0:
                hc = calculate_HC(melody.split_note_info[0])
                results['harmonic_consistency'] = hc
                results['harmonic_consistency_mean'] = np.mean(hc)
            else:
                results['harmonic_consistency'] = []
                results['harmonic_consistency_mean'] = np.nan

            all_results.append(
                results
            )

        del melody

    now = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

    json.dump(errors,
              open(f'../../data/alignment_scores/v{version}/errors-{now}.json',
                   'w+'))
    # plt.savefig(f'../../data/alignment_scores/v{version}/harmonic_consistency-{now}.png')

    res = pd.DataFrame().from_dict(all_results)
    res.to_csv(f'../../data/alignment_scores/v{version}/results-{now}.csv')
