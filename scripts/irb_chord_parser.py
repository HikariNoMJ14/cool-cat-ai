import re
import json
from jchord.progressions import ChordProgression, Song, SongSection

irb_files = json.load(open('data/iRb_v1-0/irb.json'))

add_chord_prog = {}

for file in irb_files:
    parts = file['sections']
    chord_prog = {}

    title = file['info']['title']

    for part, chords in file['content'].items():
        chord_list = []
        for chord_info in chords:
            chord = chord_info['chord']
            chord = re.sub(r'([A-G][b#]*)ø(7)', r'\1m\2b5', chord)
            chord = re.sub(r'([A-G][b#]*)o(7)', r'\1dim\2', chord)

            chord_list.append(chord)

            for i in range(chord_info['duration']['beats'] - 1):
                chord_list.append('--')

        chord_prog[part] = " ".join(chord_list)

    add_chord_prog[title] = chord_prog

json.dump('./data/chord_progressions/')
