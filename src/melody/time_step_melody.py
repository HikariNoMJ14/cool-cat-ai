import os
import pandas as pd
import numpy as np
import pretty_midi as pm
import torch

from src.melody import Melody
from src.utils import is_weakly_polyphonic, is_strongly_polyphonic, \
                      remove_weak_polyphony, remove_strong_polyphony, \
                      flatten_chord_progression
from src.utils.ezchord import Chord
from src.utils.constants import REST_SYMBOL

dir_path = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.join(dir_path, '..', '..')

# TODO Class inheritance not great - rethink


class TimeStepMelody(Melody):
    VERSION = '1.2'

    def __init__(self, filepath, polyphonic, chord_encoding_type, chord_extension_count, duration_correction):
        super(TimeStepMelody, self).__init__(filepath, self.VERSION, chord_encoding_type, chord_extension_count)

        self.encoded = None
        self.polyphonic = polyphonic
        self.encoded_folder = os.path.join(
            src_path,
            'data',
            'encoded',
            'timestep',
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
        if improvised_filepath is not None:
            improvised = pd.read_csv(improvised_filepath, index_col=0)
            improvised['end_ticks'] = improvised['ticks'] + improvised['duration']
            improvised.sort_values('ticks', inplace=True)

            if is_strongly_polyphonic(improvised):
                improvised = remove_strong_polyphony(improvised)

            if is_weakly_polyphonic(improvised):
                improvised = remove_weak_polyphony(improvised)

        original = pd.read_csv(original_filepath, index_col=0)
        original['end_ticks'] = original['ticks'] + original['duration']
        original.sort_values('ticks', inplace=True)

        if is_strongly_polyphonic(original):
            original = remove_strong_polyphony(original)

        if is_weakly_polyphonic(original):
            original = remove_weak_polyphony(original)

        flat_chord_progression = flatten_chord_progression(self.song_structure)
        num_chord_progression_beats = int(len(flat_chord_progression))

        n_ticks = self.FINAL_TICKS_PER_BEAT * num_chord_progression_beats

        rows = []
        for i in range(n_ticks):
            offset = i % (self.FINAL_TICKS_PER_BEAT * self.chord_progression_time_signature[0])

            original_pitch = np.nan
            original_pitches = original[
                (original['ticks'] <= i) & (i < original['end_ticks'])
                ]

            if len(original_pitches) > 0:
                original_pitch = original_pitches['pitch'].values[0]

                if len(original_pitches) > 1:
                    print(original_pitches)
                    raise Exception('Error!!! not mono pitch on original')

            original_attack = 0
            original_attacks = original[original['ticks'] == i]

            if len(original_attacks) > 0:
                original_attack = 1

                if len(original_attacks) > 1:
                    raise Exception('Error!!! not mono attack on original')

            improvised_pitch = np.nan
            improvised_attack = 0

            if improvised_filepath is not None:

                improvised_pitches = improvised[
                    (improvised['ticks'] <= i) & (i < improvised['end_ticks'])
                    ]

                if len(improvised_pitches) > 0:
                    improvised_pitch = improvised_pitches['pitch'].values[0]

                    if len(improvised_pitches) > 1:
                        raise Exception('Error!!! not mono pitch on improvised')

                improvised_attacks = improvised[improvised['ticks'] == i]

                if len(improvised_attacks) > 0:
                    improvised_attack = 1

                    if len(improvised_attacks) > 1:
                        raise Exception('Error!!! not mono attack on improvised')

            chord_name = flat_chord_progression[(i // self.FINAL_TICKS_PER_BEAT)]

            rows.append({
                'offset': offset,
                'improvised_pitch': improvised_pitch,
                'improvised_attack': improvised_attack,
                'original_pitch': original_pitch,
                'original_attack': original_attack,
                'chord_name': chord_name
            })

        # TODO gather statistics on pitch distribution and normalize based on vocal ranges
        # self.normalize_pitch_range(self.encoded)

        return pd.DataFrame(rows)

    # TODO move to subclass?
    def encode_poly(self, improvised_filepath, original_filepath):
        improvised = pd.read_csv(improvised_filepath, index_col=0)
        improvised['end_ticks'] = improvised['ticks'] + improvised['duration']

        original = pd.read_csv(original_filepath, index_col=0)
        original['end_ticks'] = original['ticks'] + original['duration']

        flat_chord_progression = flatten_chord_progression(self.song_structure)
        num_chord_progression_beats = int(len(flat_chord_progression))

        n_ticks = self.FINAL_TICKS_PER_BEAT * num_chord_progression_beats

        rows = []
        for i in range(n_ticks):
            offset = i % (self.FINAL_TICKS_PER_BEAT * self.chord_progression_time_signature[0])

            improvised_sustains = improvised[
                (improvised['ticks'] < i) & (i < improvised['end_ticks'])
                ]['pitch'].values
            improvised_sustains = self.multiple_pitches_to_string(improvised_sustains)

            improvised_attacks = improvised[improvised['ticks'] == i]['pitch'].values
            improvised_attacks = self.multiple_pitches_to_string(improvised_attacks)

            original_sustains = original[
                (original['ticks'] < i) & (i < original['end_ticks'])
                ]['pitch'].values
            original_sustains = self.multiple_pitches_to_string(original_sustains)

            original_attacks = original[original['ticks'] == i]['pitch'].values
            original_attacks = self.multiple_pitches_to_string(original_attacks)

            chord_name = flat_chord_progression[(i // self.FINAL_TICKS_PER_BEAT)]

            rows.append({
                'offset': offset,
                'improvised_sustains': improvised_sustains,
                'improvised_attacks': improvised_attacks,
                'original_sustains': original_sustains,
                'original_attacks': original_attacks,
                'chord_name': chord_name
            })

            return pd.DataFrame(rows)

    def save_encoded(self):
        if not os.path.exists(f'{self.encoded_folder}/{self.source}'):
            os.makedirs(f'{self.encoded_folder}/{self.source}')

        self.encoded.to_csv(f'{self.encoded_folder}/{self.source}/{self.filename}')

    def to_tensor(self, transpose_interval):
        offsets = torch.from_numpy(
            np.array([self.encoded['offset']])
        ).long().clone()

        improvised_pitches = torch.from_numpy(
            np.array((self.encoded[['improvised_pitch']] + transpose_interval).fillna(REST_SYMBOL))
        ).long().clone().transpose(0, 1)

        improvised_attacks = torch.from_numpy(
            np.array(self.encoded[['improvised_attack']])
        ).long().clone().transpose(0, 1)

        original_pitches = torch.from_numpy(
            np.array([(self.encoded['original_pitch'] + transpose_interval).fillna(REST_SYMBOL)])
        ).long().clone()

        original_attacks = torch.from_numpy(
            np.array([self.encoded['original_attack']])
        ).long().clone()

        chord_pitches = torch.from_numpy(
            np.stack(
                self.encoded['chord_name'].apply(
                    lambda x: self.transpose_chord(x, transpose_interval)
                ).fillna(REST_SYMBOL)
            )
        ).long().clone().transpose(0, 1)

        return torch.cat([
            offsets,
            improvised_pitches,
            improvised_attacks,
            original_pitches,
            original_attacks,
            chord_pitches
        ], 0).transpose(0, 1)[None, :, :]

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
            0
        )

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

        attack_df = self.encoded[self.encoded['improvised_attack'] == True]
        attack_df.reset_index(inplace=True)

        for i, row in attack_df.iterrows():
            start = row.ticks * multiplier

            duration = 0

            next_idx = min(len(attack_df) - 1, i + 1)
            next_att = attack_df.iloc[next_idx]['ticks']

            for j, row2 in self.encoded.iloc[row.ticks:next_att].iterrows():
                if row2['improvised_pitch'] == row['improvised_pitch']:
                    duration += 1

            end = start + duration * multiplier

            note = pm.Note(
                velocity=127,
                pitch=int(row["improvised_pitch"]),
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
