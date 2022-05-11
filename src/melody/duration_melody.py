import os
import json
import numpy as np
import pandas as pd
import torch
import pretty_midi as pm

from src.melody import Melody
from src.ezchord import Chord
from src.utils import get_chord_progressions, is_weakly_polyphonic, is_strongly_polyphonic, \
    remove_weak_polyphony, remove_strong_polyphony, \
    flatten_chord_progression
from src.utils.constants import OCTAVE_SEMITONES

dir_path = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.join(dir_path, '..', '..')

chord_mapping_filepath = os.path.join(src_path, 'data', 'tensor_dataset', 'chords', 'extended_7.json')


class DurationMelody(Melody):
    VERSION = '1.2'

    def __init__(self, filepath, polyphonic, duration_correction):
        super(DurationMelody, self).__init__(filepath, self.VERSION)

        self.encoded = None
        self.polyphonic = polyphonic
        self.encoded_folder = os.path.join(
            src_path,
            'data',
            'encoded',
            'duration',
            'poly' if self.polyphonic else 'mono',
        )
        self.duration_correction = duration_correction

        # TODO pass chord encoding type
        with open(chord_mapping_filepath) as fp:
            self.chord_mapping = json.load(fp)

    @staticmethod
    def multiple_pitches_to_string(pitches):
        return "-".join(str(x) for x in pitches)

    def encode(self, improvised_filepath, original_filepath):
        if self.polyphonic:
            encoded = self.encode_poly(improvised_filepath, original_filepath)
        else:
            encoded = self.encode_mono(improvised_filepath, original_filepath)

        self.encoded = encoded

    def encode_mono(self, improvised_filepath, original_filepath):
        encoded = []

        if improvised_filepath is not None:
            improvised = pd.read_csv(improvised_filepath, index_col=0).reset_index(drop=True)
            improvised['end_ticks'] = improvised['ticks'] + improvised['duration']
            improvised.sort_values('ticks', inplace=True)

            if is_strongly_polyphonic(improvised):
                improvised = remove_strong_polyphony(improvised)

            if is_weakly_polyphonic(improvised):
                improvised = remove_weak_polyphony(improvised)

            encoded.append(self.create_encoded_dataset(improvised, 'improvised'))

        original = pd.read_csv(original_filepath, index_col=0).reset_index(drop=True)
        original['end_ticks'] = original['ticks'] + original['duration']
        original.sort_values('ticks', inplace=True)

        if is_strongly_polyphonic(original):
            original = remove_strong_polyphony(original)

        if is_weakly_polyphonic(original):
            original = remove_weak_polyphony(original)

        encoded.append(self.create_encoded_dataset(original, 'original'))

        return pd.concat(encoded, axis=0)

    def create_encoded_dataset(self, dataset, phase):
        rows = []
        bpm = self.chord_progression_time_signature[0]
        ftpm = self.FINAL_TICKS_PER_BEAT * bpm
        flat_chord_progression = flatten_chord_progression(self.song_structure)
        ticks = 0
        duration = 0

        def get_current_chord(row):
            offset = row['measure'] + (row['offset'] / ftpm)
            chord_idx = int(np.floor(offset * bpm))

            return flat_chord_progression[chord_idx]

        for i, row in dataset.iterrows():
            diff = row['ticks'] - (duration + ticks)

            if diff > self.duration_correction:
                rows.append({
                    'ticks': (ticks + duration),
                    'offset': (ticks + duration) % ftpm,
                    'pitch': np.nan,
                    'duration': diff,
                    'chord_name': get_current_chord(row),
                    'type': phase
                })

            ticks = row['ticks']
            duration = row['duration']

            if i + 1 < dataset.shape[0]:
                next_diff = dataset.iloc[i + 1]['ticks'] - (row['duration'] + row['ticks'])
                if next_diff <= self.duration_correction:
                    duration = duration + next_diff

            rows.append({
                'ticks': row['ticks'],
                'offset': row['offset'],
                'pitch': row['pitch'],
                'duration': duration,
                'chord_name': row['chord_name'],
                'type': phase
            })

        encoded = pd.DataFrame(rows)
        self.validate_encoded_dataset(encoded)

        return encoded

    def validate_encoded_dataset(self, encoded):
        for i in range(encoded.shape[0] - 1):
            current_row = encoded.iloc[i]

            current_ticks = current_row['ticks']
            current_duration = current_row['duration']

            next_row = encoded.iloc[i + 1]
            next_ticks = next_row['ticks']

            if int(current_ticks + current_duration) != int(next_ticks):
                raise Exception(f'Failed validation: {self.filename}')

    def encode_poly(self, i, o):
        return False

    def save_encoded(self):
        if not os.path.exists(f'{self.encoded_folder}/{self.source}'):
            os.makedirs(f'{self.encoded_folder}/{self.source}')

        self.encoded.to_csv(f'{self.encoded_folder}/{self.source}/{self.filename}')

    # TODO choice for which extensions to add is a bit random
    @staticmethod
    def extended_chord_encoding(chord_pitches, chord_notes_count):
        upper_extension_index = -3

        # If 13th chord, remove 11th
        if len(chord_pitches) == 8:
            del chord_pitches[-2]

        while True:
            if len(chord_pitches) == chord_notes_count:
                # if chord_pitches != sorted(chord_pitches):
                #     print(f'Notes {chord_pitches} not sorted')

                return np.array(sorted(chord_pitches))

            # Append extensions to fill up 'chord_notes_count' notes
            upper_extension = chord_pitches[upper_extension_index] + OCTAVE_SEMITONES
            chord_pitches.append(upper_extension)

    def to_tensor(self, transpose_interval):
        improvised_encoded = self.encoded[self.encoded['type'] == 'improvised']

        improvised_offsets = torch.from_numpy(
            np.array([improvised_encoded['offset']])
        ).long().clone()

        improvised_pitches = torch.from_numpy(
            np.array((improvised_encoded[['pitch']] + transpose_interval).fillna(128))
        ).long().clone().transpose(0, 1)

        improvised_duration = torch.from_numpy(
            np.array(improvised_encoded[['duration']])
        ).long().clone().transpose(0, 1)

        improvised_chord_pitches = torch.from_numpy(
            np.stack(
                improvised_encoded['chord_name'].apply(
                    lambda x: np.array(self.chord_mapping[x]) + transpose_interval
                )
            )
        ).long().clone().transpose(0, 1)

        improvised_tensor = torch.cat([
            improvised_offsets,
            improvised_pitches,
            improvised_duration,
            improvised_chord_pitches
        ], 0).transpose(0, 1)

        original_encoded = self.encoded[self.encoded['type'] == 'original']

        original_offsets = torch.from_numpy(
            np.array([original_encoded['offset']])
        ).long().clone()

        original_pitches = torch.from_numpy(
            np.array([(original_encoded['pitch'] + transpose_interval).fillna(128)])
        ).long().clone()

        original_duration = torch.from_numpy(
            np.array([original_encoded['duration']])
        ).long().clone()

        original_chord_pitches = torch.from_numpy(
            np.stack(
                original_encoded['chord_name'].apply(
                    lambda x: np.array(self.chord_mapping[x]) + transpose_interval
                )
            )
        ).long().clone().transpose(0, 1)

        original_tensor = torch.cat([
            original_offsets,
            original_pitches,
            original_duration,
            original_chord_pitches
        ], 0).transpose(0, 1)

        return (
            improvised_tensor,
            original_tensor
        )

    def to_midi(
            self,
            out_filepath: str,
            out_bpm: int = 120,
    ):
        melody_instrument_name = "Tenor Sax"
        chord_instrument_name = "Acoustic Grand Piano"

        p = pm.PrettyMIDI()
        ts = pm.TimeSignature(
            self.chord_progression_time_signature[0],
            self.chord_progression_time_signature[1],
            0)

        p.time_signature_changes.append(ts)

        melody = pm.Instrument(
            program=pm.instrument_name_to_program(melody_instrument_name),
            name="melody"
        )
        chords = pm.Instrument(
            program=pm.instrument_name_to_program(chord_instrument_name),
            name="chords"
        )

        multiplier = (out_bpm / 60) / (self.FINAL_TICKS_PER_BEAT * self.chord_progression_time_signature[0])

        self.encoded['ticks'] = self.encoded.index

        notes_df = self.encoded[~np.isnan(self.encoded['improvised_pitch'])]
        notes_df.reset_index(inplace=True)

        for i, row in notes_df.iterrows():
            duration = row['improvised_duration']
            start = row.ticks * multiplier
            end = start + duration * multiplier

            note = pm.Note(
                velocity=127,
                pitch=int(row['improvised_pitch']),
                start=start,
                end=end,
            )
            melody.notes.append(note)

        p.instruments.append(melody)

        start = 0
        use_tonic = True

        for section in self.song_structure['sections']:
            for chord_name in self.song_structure['progression'][section]:
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

        p.write(out_filepath)


if __name__ == "__main__":
    chord_progressions = get_chord_progressions(src_path)

    d = DurationMelody(
        filepath='../data/split_melody_data/v1.2/JazzPage/A Felicidade -1-.csv',
        polyphonic=False,
        duration_correction=2
    )
    d.set_song_structure(chord_progressions[d.song_name])

    d.encode(
        # improvised_filepath=None,
        improvised_filepath='../../data/split_melody_data/v1.2/JazzPage/A Felicidade -1-.csv',
        original_filepath='../../data/split_melody_data/v1.2/Real Book/A Felicidade -o-.csv',
    )
    d.save_encoded()
