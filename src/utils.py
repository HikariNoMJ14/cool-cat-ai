import os
import copy

import pandas as pd
import numpy as np

from ezchord import Chord

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

    out_path = os.path.join('..', 'data', 'Complete Examples Melodies Manual')
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


def midi_to_notes(midi_file: str) -> pd.DataFrame:
    p = pm.PrettyMIDI(midi_file)
    instrument = p.instruments[0]
    notes = pm.collections.defaultdict(list)

    # Sort the notes by start time
    sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
    prev_start = sorted_notes[0].start
    init_time = sorted_notes[0].start

    for note in sorted_notes:
        start = note.start - init_time
        end = note.end - init_time
        notes['pitch'].append(note.pitch)
        notes['start'].append(start)
        notes['end'].append(end)
        notes['velocity'].append(note.velocity)
        notes['step'].append(start - prev_start)
        notes['duration'].append(end - start)
        prev_start = start
    notes["step"][0] = 0

    return pd.DataFrame({name: np.array(value) for name, value in notes.items()})


def notes_to_midi(notes: pd.DataFrame,
                  out_file: str,
                  instrument_name: str = "Acoustic Grand Piano") -> pm.PrettyMIDI:

    p = pm.PrettyMIDI()
    instrument = pm.Instrument(
        program=pm.instrument_name_to_program(instrument_name)
    )

    for i, note in notes.iterrows():
        note = pm.Note(
            velocity=127,
            pitch=int(note["pitch"]),
            start=note["ticks"] / 48,
            end=(note["ticks"] + note['duration']) / 48,
        )
        instrument.notes.append(note)

    p.instruments.append(instrument)
    p.write(out_file)

    return p


def notes_and_chord_to_midi(
        notes: pd.DataFrame,
        chord_progression: dict,
        out_file: str) -> pm.PrettyMIDI:
    instrument_name = "Acoustic Grand Piano"

    p = pm.PrettyMIDI()
    melody = pm.Instrument(
        program=pm.instrument_name_to_program(instrument_name),
        name="melody"
    )
    chords = pm.Instrument(
        program=pm.instrument_name_to_program(instrument_name),
        name="chords"
    )

    for i, note in notes.iterrows():
        note = pm.Note(
            velocity=127,
            pitch=int(note["pitch"]),
            start=note["ticks"] / 24,
            end=(note["ticks"] + note['duration']) / 24,
        )
        melody.notes.append(note)

    p.instruments.append(melody)

    start = 0
    use_tonic = True

    for section in chord_progression['sections']:
        for chord in chord_progression['progression'][section]:
            chord_notes = Chord(chord).getMIDI()

            if use_tonic:
                note = pm.Note(
                    velocity=64,
                    pitch=int(chord_notes[0]),
                    start=start,
                    end=start + 0.5,
                )
                chords.notes.append(note)
            else:
                for chord_note in chord_notes[1:]:
                    note = pm.Note(
                        velocity=64,
                        pitch=int(chord_note),
                        start=start,
                        end=start + 0.5,
                    )
                    chords.notes.append(note)

            start += 0.5
            use_tonic = not use_tonic

    p.instruments.append(chords)

    p.write(out_file)

    return p

# if __name__ == "__main__":
# for i in range(128):
#     print(int_to_pitch(i))
