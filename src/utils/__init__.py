# TODO move to own files
import os
import re
from glob import glob
import copy
import json

import pandas as pd
import numpy as np

from src.ezchord import Chord

import pretty_midi as pm
import mingus.core.notes as notes

from src.objective_metrics import calculate_HC, calculate_silence_ratio
from src.utils.metrics import Metric
from src.utils.constants import SOURCES, INPUT_DATA_FOLDER

dir_path = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.join(dir_path, '..', '..')

melody_names = ['melody', 'melodia', 'melodía', 'lead']
non_melody_names = ['bass', 'bajo', 'basso', 'baixo', 'drum', 'percussion', 'batería', 'bateria', 'chord', 'rhythm',
                    'cymbal', 'clap', 'kick', 'snare', 'hh ', 'hats', 'ride', 'kit']

bass_programs = list(range(32, 40))
ensemble_programs = list(range(48, 56))
pad_programs = list(range(88, 96))
synth_effect_programs = list(range(96, 104))
percussive_programs = list(range(112, 120))
sound_effect_programs = list(range(120, 128))

non_melody_programs = bass_programs + \
                      ensemble_programs + \
                      pad_programs + \
                      synth_effect_programs + \
                      percussive_programs + \
                      sound_effect_programs


def get_filepaths(mode):
    filepaths = []
    for source in SOURCES[mode]:
        filepaths += [y for x in os.walk(os.path.join(INPUT_DATA_FOLDER, source))
                      for y in glob(os.path.join(x[0], '*.csv'))]
    return filepaths


def get_original_filepath(song_name):
    original_filepaths = [filepath for filepath in get_filepaths('original')
                          if filepath_to_song_name(filepath) == song_name]

    if len(original_filepaths) > 1:
        raise Exception(f'Multiple original files match the song name {song_name}, {original_filepaths}')

    return original_filepaths[0]


# not working properly
def find_chords(filename):
    from omnizart.chord import app as capp
    from omnizart.utils import synth_midi

    synth_filename = filename.replace('standardized', 'synthesized').replace('mid', 'wav')

    synth_midi(filename, synth_filename)

    output_filename = filename.replace('standardized', 'chords')

    capp.transcribe(synth_filename, output=output_filename)

    return output_filename


def filepath_to_song_name(filepath):
    song_name = os.path.basename(filepath).replace('.mid', '').replace('.csv', '')
    song_name = "".join(song_name.split(' - ')[-1])
    song_name = re.sub('-[0-9,o]*-', '', song_name)
    song_name = re.sub('\(.*\)', '', song_name).strip()

    return song_name

def filename_to_songname(filename):
    songname = filename.replace('.mid', '')
    songname = re.sub(r' \([0-9]\)', '', songname)
    songname = re.sub(r'.* - (.*)', r'\1', songname)

    return songname


def int_to_pitch(num):
    pitch = notes.int_to_note(num % 12, 'b')

    if 'G' in pitch:
        pitch = notes.int_to_note(num % 12, '#')

    octave = (num // 12) - 1

    return f"{pitch}{octave}"


def filter_instruments_old(pm_p):
    filtered_instr = [
        i for i in pm_p.instruments
        if not i.is_drum
           and all(
            sub not in i.name.lower() for sub in non_melody_names
        )]

    return filtered_instr


def filter_instruments_new(pm_p):
    filtered_instr = [
        i for i in pm_p.instruments
        if not i.is_drum
           and i.program not in non_melody_programs
           and all(sub not in i.name.lower() for sub in non_melody_names)
    ]

    return filtered_instr


def get_melody_tracks(pm_p):
    melody_tracks = [i for i in pm_p.instruments if 'solo' in i.name.lower()]

    if len(melody_tracks) == 0:
        melody_tracks = [
            i for i in pm_p.instruments
            if any(
                sub in i.name.lower() for sub in melody_names
            )]

    return melody_tracks


def extract_melody_by_index(file, melody_idx, out_path):
    pm_m = pm.PrettyMIDI(file)

    source = os.path.basename(os.path.dirname(file))

    filtered_instr = filter_instruments_old(pm_m)
    melody_track = filtered_instr[melody_idx]

    if not os.path.exists(os.path.join(out_path, source)):
        os.makedirs(os.path.join(out_path, source))

    out_filename = os.path.join(out_path, source, os.path.basename(file))

    pm_melody = copy.deepcopy(pm_m)
    pm_melody.instruments = [melody_track]

    pm_melody.write(out_filename)


def extract_melody_by_name(file, melody_name, melody_name_idx=0):
    pm_m = pm.PrettyMIDI(file)

    out_path = os.path.join('..', 'data', 'Complete Examples Melodies Manual')
    source = os.path.basename(os.path.dirname(file))

    filtered_instr = filter_instruments_old(pm_m)

    melody_tracks = [i for i in filtered_instr if melody_name in i.name.lower()]
    melody_track = melody_tracks[melody_name_idx]

    if not os.path.exists(os.path.join(out_path, source)):
        os.mkdir(os.path.join(out_path, source))

    out_filename = os.path.join(out_path, source, os.path.basename(file))

    pm_melody = copy.deepcopy(pm_m)
    pm_melody.instruments = [melody_track]

    pm_melody.write(out_filename)


def get_chord_progressions(src_folder='../..'):
    irb_chord_progressions_filepath = os.path.join(
        src_folder, 'data/chord_progressions/irb_chord_progressions.json')
    wdb_chord_progressions_filepath = os.path.join(
        src_folder, 'data/chord_progressions/weimar_db.json')
    manual_chord_progressions_filepath = os.path.join(
        src_folder, 'data/chord_progressions/manual_chord_progressions.json')

    chord_progressions = {}
    chord_progressions.update(json.load(open(irb_chord_progressions_filepath)))
    chord_progressions.update(json.load(open(wdb_chord_progressions_filepath)))
    chord_progressions.update(json.load(open(manual_chord_progressions_filepath)))

    return chord_progressions


def flatten_chord_progression(chord_progression):
    linear_chord_progression = []

    for section in chord_progression['sections']:
        linear_chord_progression += chord_progression['progression'][section]

    # print(chord_progression['sections'])

    return linear_chord_progression


def is_weakly_polyphonic(melody):
    melody['end_ticks'] = melody['ticks'] + melody['duration']

    return melody[
               (melody['end_ticks'].shift(1) > melody['ticks']) |
               (melody['end_ticks'] > melody['ticks'].shift(-1))
               ][['ticks', 'end_ticks']].shape[0] > 0


def is_strongly_polyphonic(melody):
    return melody['ticks'].shape[0] > melody['ticks'].nunique()


def remove_weak_polyphony(melody):
    new_melody = melody.copy()

    overlap = (new_melody['end_ticks'] - new_melody['ticks'].shift(-1)).clip(0, None)

    # skip last row as 'shift' messes it up
    new_melody.iloc[:-1, new_melody.columns.get_loc('duration')] -= overlap.iloc[:-1]
    new_melody.iloc[:-1, new_melody.columns.get_loc('end_ticks')] -= overlap.iloc[:-1]

    if is_weakly_polyphonic(new_melody):
        raise Exception('Error!!! Weak polyphony not removed correctly')

    return new_melody


def remove_strong_polyphony(melody):
    new_melody = melody.copy()

    new_melody = new_melody\
        .sort_values('pitch', ascending=False)\
        .drop_duplicates('ticks')\
        .sort_values('ticks')

    if is_strongly_polyphonic(new_melody):
        raise Exception('Error!!! Strong polyphony not removed correctly')

    return new_melody


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
        quantized: bool,
        out_file: str) -> pm.PrettyMIDI:
    melody_instrument_name = "Tenor Sax"
    chord_instrument_name = "Acoustic Grand Piano"

    #     print(notes)

    p = pm.PrettyMIDI()
    melody = pm.Instrument(
        program=pm.instrument_name_to_program(melody_instrument_name),
        name="melody"
    )
    chords = pm.Instrument(
        program=pm.instrument_name_to_program(chord_instrument_name),
        name="chords"
    )

    if quantized:
        ticks_col = "quant_ticks"
        durations_col = "quant_duration"

        multiplier = 1 / 24
    else:
        ticks_col = "raw_ticks"
        durations_col = "raw_duration"

        multiplier = 1

    for i, note in notes.iterrows():
        start = note[ticks_col] * multiplier
        end = (note[ticks_col] + note[durations_col]) * multiplier

        note = pm.Note(
            velocity=127,
            pitch=int(note["pitch"]),
            start=start,
            end=end,
        )
        melody.notes.append(note)

    p.instruments.append(melody)

    start = 0
    use_tonic = True

    for section in chord_progression['sections']:
        for chord_name in chord_progression['progression'][section]:
            chord_notes = Chord(chord_name).getMIDI()

            if use_tonic:
                note = pm.Note(
                    velocity=64,
                    pitch=int(chord_notes[0]),
                    start=start,
                    end=start + 0.25,
                )
                chords.notes.append(note)

                # Add chord annotation
                chord_annotation = pm.Lyric(chord_name, start)

                p.lyrics.append(chord_annotation)
            else:
                for chord_note in chord_notes[1:]:
                    note = pm.Note(
                        velocity=64,
                        pitch=int(chord_note),
                        start=start,
                        end=start + 0.25,
                    )
                    chords.notes.append(note)

            start += 0.5
            use_tonic = not use_tonic

    p.instruments.append(chords)

    p.write(out_file)

    return p


def calculate_melody_results(melody):
    all_results = {}
    for i, melody_info in enumerate(melody.split_note_info):
        results = melody.chord_progression_comparison()
        results.update({
            'source': melody.source,
            'in_filename': f'{melody.filename}',
            'out_filename': f'{melody.filename.replace(".mid", "")} -{i + 1 if not melody.original else "o"}-.mid',
            'starting_measure': melody.starting_measure,
            'melody_mido_key': melody.mido_key,
            'chord_progression_key': melody.chord_progression_key + 'm'
            if melody.chord_progression_minor
            else melody.chord_progression_key,
            'transpose_semitones': melody.transpose_semitones
        })

        sr = calculate_silence_ratio(melody_info)
        results['silence_ratio'] = sr

        if len(melody.split_note_info) > 0:
            hc = calculate_HC(melody_info)
            results['harmonic_consistency'] = hc
            results['harmonic_consistency_mean'] = np.mean(hc)
            results['harmonic_consistency_var'] = np.var(hc)
        else:
            results['harmonic_consistency'] = []
            results['harmonic_consistency_mean'] = np.nan
            results['harmonic_consistency_var'] = np.nan

        all_results[i] = results

    return all_results

# if __name__ == "__main__":
# for i in range(128):
#     print(int_to_pitch(i))
