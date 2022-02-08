import os
import copy

import pretty_midi as pm
import mingus.core.notes as notes

melody_names = ['melody', 'melodia', 'melodía', 'lead']
non_melody_names = ['bass', 'bajo', 'basso', 'baixo', 'drum', 'percussion', 'batería', 'bateria', 'chord', 'rhythm',
                    'cymbal', 'clap', 'kick', 'snare', 'hh ', 'hats', 'ride', 'kit']


def int_to_pitch(num):
    pitch = notes.int_to_note(num % 12, 'b')

    if 'G' in pitch:
        pitch = notes.int_to_note(num % 12, '#')

    octave = (num // 12) - 1

    return f"{pitch}{octave}"


def filter_instruments(pm_p):
    filtered_instr = [
        i for i in pm_p.instruments
        if i.is_drum == False
           and all(
            sub not in i.name.lower() for sub in non_melody_names
        )]

    return filtered_instr


def extract_melody_by_index(file, melody_idx):
    pm_m = pm.PrettyMIDI(file)

    out_path = os.path.join('..', 'data', 'Complete Examples Melodies Secondary')
    source = os.path.basename(os.path.dirname(file))

    filtered_instr = filter_instruments(pm_m)
    melody_track = filtered_instr[melody_idx]

    if not os.path.exists(os.path.join(out_path, source)):
        os.mkdir(os.path.join(out_path, source))

    out_filename = os.path.join(out_path, source, os.path.basename(file))

    pm_melody = copy.deepcopy(pm_m)
    pm_melody.instruments = [melody_track]

    print(melody_track.name)

    pm_melody.write(out_filename)


def extract_melody_by_name(file, melody_name, melody_name_idx=0):
    pm_m = pm.PrettyMIDI(file)

    out_path = os.path.join('..', 'data', 'Complete Examples Melodies Manual')
    source = os.path.basename(os.path.dirname(file))

    filtered_instr = filter_instruments(pm_m)

    melody_tracks = [i for i in filtered_instr if melody_name in i.name.lower()]
    melody_track = melody_tracks[melody_name_idx]

    if not os.path.exists(os.path.join(out_path, source)):
        os.mkdir(os.path.join(out_path, source))

    out_filename = os.path.join(out_path, source, os.path.basename(file))

    pm_melody = copy.deepcopy(pm_m)
    pm_melody.instruments = [melody_track]

    print(melody_track.name)

    pm_melody.write(out_filename)


# if __name__ == "__main__":
# for i in range(128):
#     print(int_to_pitch(i))
