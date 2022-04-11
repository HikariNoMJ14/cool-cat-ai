import os
import json
import re
from glob import glob
from datetime import datetime

import music21.key
import pandas as pd

from melody.melody import Melody
from utils import get_chord_progressions, calculate_melody_results

if __name__ == "__main__":
    version = '1.2'
    quantized = True

    wdb_info = pd.read_csv('../../data/sources/WeimarDB/weimar_db_melody_info_with_pitch.csv')

    folder = f'../../data/Complete Examples Melodies Auto/v{version}/Weimar DB'

    filepaths = [y for x in os.walk(folder) for y in glob(os.path.join(x[0], '*.mid'))]

    filepaths = [
        '../../data/Complete Examples Melodies Auto/v1.2/Weimar DB/Sidney Bechet - Summertime.mid'
    ]

    all_results = []
    errors = {}

    cp = get_chord_progressions()

    starting_measure_exceptions = {
        'Weimar DB - Charlie Parker - Ko-Ko': 33,
        'Weimar DB - Lee Konitz - Mean to Me': 17,
        'Weimar DB - Zoot Sims - Night And Day': 17
    }

    for f in filepaths:
        melody = Melody(f, version)
        dict_key = os.path.join(melody.source, melody.filename)

        print(os.path.basename(f))

        a = [
            x.strip() for x in re.sub(
                r'-[0-9]+-', '',
                os.path.basename(melody.filename).replace('.mid', '')
            ).split('-')
        ]

        perf = a[0]
        songname = a[1]

        melody.setup()
        melody.parse_notes()
        melody.set_song_structure(cp[melody.song_name])

        if len(melody.errors) == 0:
            song_wdb_info = wdb_info[
                (wdb_info['performer'] == perf) &
                (wdb_info['title'] == songname)
            ]
            # TODO - change to bar where you find the first A1 form
            one_bar = song_wdb_info[song_wdb_info['bar'] == 1]

            if len(one_bar['pitch'].dropna()) > 0:
                try:
                    keys = one_bar['key'].unique()

                    info_key = keys[0].replace('-', '')

                    new_key = info_key.replace('maj', '').replace('min', 'm')
                    melody.key = music21.key.Key(new_key)
                except:
                    pass

                starting_note_idx = one_bar['pitch'].dropna().index[0]
                notes_up_to_start = len(song_wdb_info.loc[:starting_note_idx]['pitch'].dropna()) - 1
                starting_measure = int(melody.note_info.iloc[notes_up_to_start]['measure'])

                # info_chords = set(one_bar['chord'].dropna())

                # info_tempo = song_wdb_info['avgtempo'].unique()[0]
                # song_tempo = 60000000 / melody.tempo
                #
                # if int(round(info_tempo)) != int(round(song_tempo)):
                #     print('Different tempo!!!', info_tempo, song_tempo)
                #
                # one_bar.loc[:, 'beat_onset'] = one_bar.loc[:, 'onset'] * (info_tempo / 60)
                #
                # starting_measure = int(np.floor(one_bar.iloc[0]['beat_onset'] / 4))

                if f'{melody.source} - {melody.song_name}' in starting_measure_exceptions:
                    starting_measure = starting_measure_exceptions[f'{melody.source} - {melody.song_name}']
                    print(f'Using {starting_measure} for {melody.source} - {melody.song_name}')

                melody.manually_align(starting_measure, quantized)
            else:
                melody.errors.append('Weimar DB - No bar 1')

        if len(melody.errors) > 0:
            errors[dict_key] = melody.errors
        else:
            all_results += calculate_melody_results(melody)

        print('---------------')

    now = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

    if len(filepaths) > 1:
        json.dump(
            errors,
            open(f'../../data/alignment_scores/v{version}/errors-{now}.json', 'w+')
        )
        # plt.savefig(f'../../data/alignment_scores/v{version}/harmonic_consistency-{now}.png')

        res = pd.DataFrame().from_dict(all_results)
        res.to_csv(f'../../data/alignment_scores/v{version}/results-{now}.csv')
