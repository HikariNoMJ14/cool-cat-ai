import re
import json

irb_files = json.load(open('/media/manu/Data/PycharmProjects/thesis/data/chord_progression_sources/iRb_v1-0/irb.json'))

add_chord_prog = {}

for file in irb_files:
    chord_prog = {
        'sections': file['sections'],
        'progression': {}
    }

    title = file['info']['title']
    key = file['info']['key']
    minor = file['info']['minor']

    chord_prog['key'] = key
    chord_prog['minor'] = minor

    print(key, minor)

    for part, chords in file['content'].items():
        chord_list = []
        for chord_info in chords:
            chord = chord_info['chord']
            chord = re.sub(r'([A-G][b#]*)ø(7)', r'\1m\2b5', chord)
            chord = re.sub(r'([A-G][b#]*)o(7)', r'\1dim\2', chord)

            chord_list.append(chord)

            for i in range(chord_info['duration']['beats'] - 1):
                chord_list.append(chord)

        chord_prog['progression'][part] = chord_list

    add_chord_prog[title] = chord_prog

json.dump(add_chord_prog, open('../data/chord_progressions/irb_chord_progressions.json', 'w+'))
