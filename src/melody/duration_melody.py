import os
import numpy as np
import pandas as pd
import torch
import pretty_midi as pm

from src.melody import Melody
from src.utils import is_weakly_polyphonic, is_strongly_polyphonic, \
    remove_weak_polyphony, remove_strong_polyphony, \
    flatten_chord_progression
from src.utils.ezchord import Chord
from src.utils.constants import OCTAVE_SEMITONES, REST_SYMBOL

dir_path = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.join(dir_path, '../../data', '..')


class DurationMelody(Melody):
    VERSION = '1.2'

    def __init__(self, filepath, polyphonic=False,
                 chord_encoding_type='extended', chord_extension_count=7, duration_correction=0):
        super(DurationMelody, self).__init__(filepath, self.VERSION, chord_encoding_type, chord_extension_count)

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

            encoded.append(self.encode_melody(improvised, 'improvised'))

        original = pd.read_csv(original_filepath, index_col=0).reset_index(drop=True)
        original['end_ticks'] = original['ticks'] + original['duration']
        original.sort_values('ticks', inplace=True)

        if is_strongly_polyphonic(original):
            original = remove_strong_polyphony(original)

        if is_weakly_polyphonic(original):
            original = remove_weak_polyphony(original)

        encoded.append(self.encode_melody(original, 'original'))

        return pd.concat(encoded, axis=0)

    def encode_melody(self, dataset, phase):
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

            # if i + 1 < dataset.shape[0]:
            #     next_diff = dataset.iloc[i + 1]['ticks'] - (row['duration'] + row['ticks'])
            #     if next_diff <= self.duration_correction:
            #         duration = duration + next_diff

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

    def to_tensor(self, transpose_interval, metadata):
        improvised_encoded = self.encoded[self.encoded['type'] == 'improvised']

        improvised_flag = torch.from_numpy(
            np.zeros((1, improvised_encoded.shape[0]))
        ).long().clone()

        improvised_ticks = torch.from_numpy(
            np.array([improvised_encoded['ticks']])
        ).long().clone()

        improvised_offsets = torch.from_numpy(
            np.array([improvised_encoded['offset']])
        ).long().clone()

        improvised_pitches = torch.from_numpy(
            np.array((improvised_encoded[['pitch']] + transpose_interval).fillna(REST_SYMBOL))
        ).long().clone().transpose(0, 1)

        improvised_duration = torch.from_numpy(
            np.array(improvised_encoded[['duration']])
        ).long().clone().transpose(0, 1)

        improvised_metadata = torch.from_numpy(
            np.tile(metadata, (improvised_encoded.shape[0], 1))
        ).long().clone().transpose(0, 1)

        improvised_chord_pitches = torch.from_numpy(
            np.stack(
                improvised_encoded['chord_name'].apply(
                    lambda x: self.transpose_chord(x, transpose_interval)
                ).fillna(REST_SYMBOL)
            )
        ).long().clone().transpose(0, 1)

        improvised_tensor = torch.cat([
            improvised_flag,
            improvised_ticks,
            improvised_offsets,
            improvised_pitches,
            improvised_duration,
            improvised_metadata,
            improvised_chord_pitches
        ], 0).transpose(0, 1)

        original_encoded = self.encoded[self.encoded['type'] == 'original']

        original_flag = torch.from_numpy(
            np.ones((1, original_encoded.shape[0]))
        ).long().clone()

        original_ticks = torch.from_numpy(
            np.array([original_encoded['ticks']])
        ).long().clone()

        original_offsets = torch.from_numpy(
            np.array([original_encoded['offset']])
        ).long().clone()

        original_pitches = torch.from_numpy(
            np.array([(original_encoded['pitch'] + transpose_interval).fillna(REST_SYMBOL)])
        ).long().clone()

        original_duration = torch.from_numpy(
            np.array([original_encoded['duration']])
        ).long().clone()

        original_metadata = torch.from_numpy(
            np.tile(metadata, (original_encoded.shape[0], 1))
        ).long().clone().transpose(0, 1)

        original_chord_pitches = torch.from_numpy(
            np.stack(
                original_encoded['chord_name'].apply(
                    lambda x: self.transpose_chord(x, transpose_interval)
                ).fillna(REST_SYMBOL)
            )
        ).long().clone().transpose(0, 1)

        original_tensor = torch.cat([
            original_flag,
            original_ticks,
            original_offsets,
            original_pitches,
            original_duration,
            original_metadata,
            original_chord_pitches
        ], 0).transpose(0, 1)

        return torch.cat([
            improvised_tensor,
            original_tensor
        ], 0)[None, :, :]

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
        beat_n = 1

        # TODO Generalize for melodies longer than one cycle
        for section in self.song_structure['sections']:
            for chord_name in self.song_structure['progression'][section]:
                chord_notes = Chord(chord_name).getMIDI()

                # Use the tonic on the first beat, the fifth on the third beat
                # and the full chord (minus the fifth) on beats 2 and 4
                # TODO only works with 4/4
                if beat_n == 1:
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
                elif beat_n == 3:
                    note = pm.Note(
                        velocity=64,
                        pitch=int(chord_notes[3]) - OCTAVE_SEMITONES * 2,
                        start=start,
                        end=start + 0.25,
                    )
                    chords.notes.append(note)
                else:
                    for extension, chord_note in enumerate(chord_notes[1:]):
                        if extension != 2:  # don't add the fifth
                            note = pm.Note(
                                velocity=64,
                                pitch=int(chord_note),
                                start=start,
                                end=start + 0.25,
                            )
                            chords.notes.append(note)

                start += 0.5
                beat_n = (beat_n + 1) % 4

        p.instruments.append(chords)

        p.write(out_filepath)
