import os
from glob import glob
import pandas as pd
import numpy as np
import pretty_midi as pm
import torch

from src.melody import Melody
from src.ezchord import Chord
from src.utils import is_weakly_polyphonic, is_strongly_polyphonic, flatten_chord_progression


# TODO Class inheritance not great - rethink


class TimeStepMelody(Melody):
    VERSION = '1.2'
    BASE_FOLDER = '../..'

    OCTAVE_SEMITONES = 12
    START_SYMBOL = 129
    END_SYMBOL = 130

    def __init__(self, filepath, polyphonic):
        super(TimeStepMelody, self).__init__(filepath, self.VERSION, self.BASE_FOLDER)

        self.encoded = None
        self.polyphonic = polyphonic
        self.encoded_folder = os.path.join(
            self.BASE_FOLDER,
            'data',
            'encoded',
            'timestep',
            'poly' if self.polyphonic else 'mono',
        )

    @staticmethod
    def multiple_pitches_to_string(pitches):
        return "-".join(str(x) for x in pitches)

    def encode(self, improvised_filepath, original_filepath, transpose_interval=0):
        if self.polyphonic:
            encoded = self.encode_poly(improvised_filepath, original_filepath, transpose_interval)
        else:
            encoded = self.encode_mono(improvised_filepath, original_filepath, transpose_interval)

        self.encoded = encoded

        # Save non-transposed encoded file
        if transpose_interval == 0:
            self.encoded.to_csv(f'{self.encoded_folder}/{self.source}')

    def encode_mono(self, improvised_filepath, original_filepath, transpose_interval=0):
        improvised = pd.read_csv(improvised_filepath, index_col=0)
        improvised['end_ticks'] = improvised['ticks'] + improvised['duration']
        improvised.sort_values('ticks', inplace=True)

        original = pd.read_csv(original_filepath, index_col=0)
        original['end_ticks'] = original['ticks'] + original['duration']
        original.sort_values('ticks', inplace=True)

        flat_chord_progression = flatten_chord_progression(self.song_structure)
        num_chord_progression_beats = int(len(flat_chord_progression))

        n_ticks = self.FINAL_TICKS_PER_BEAT * num_chord_progression_beats

        if is_strongly_polyphonic(improvised):
            improvised = self.remove_strong_polyphony(improvised)

        if is_weakly_polyphonic(improvised):
            improvised = self.remove_weak_polyphony(improvised)

        if is_strongly_polyphonic(original):
            original = self.remove_strong_polyphony(original)

        if is_weakly_polyphonic(original):
            original = self.remove_weak_polyphony(original)

        rows = []
        for i in range(n_ticks):
            offset = i % (self.FINAL_TICKS_PER_BEAT * self.chord_progression_time_signature[0])

            improvised_pitch = np.nan
            improvised_pitches = improvised[
                (improvised['ticks'] <= i) & (i < improvised['end_ticks'])
                ]

            if len(improvised_pitches) > 0:
                improvised_pitch = improvised_pitches['pitch'].values[0]

                if len(improvised_pitches) > 1:
                    raise Exception('Error!!! not mono pitch on improvised')

            improvised_attack = 0
            improvised_attacks = improvised[improvised['ticks'] == i]

            if len(improvised_attacks) > 0:
                improvised_attack = 1

                if len(improvised_attacks) > 1:
                    raise Exception('Error!!! not mono attack on improvised')

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

            chord_name = flat_chord_progression[(i // self.FINAL_TICKS_PER_BEAT)]

            rows.append({
                'offset': offset,
                'improvised_pitch': improvised_pitch + transpose_interval,
                'improvised_attack': improvised_attack,
                'original_pitch': original_pitch + transpose_interval,
                'original_attack': original_attack,
                'chord_name': chord_name
            })

        # TODO gather statistics on pitch distribution and normalize based on vocal ranges
        # self.normalize_pitch_range(self.encoded)

        return pd.DataFrame(rows)

    # TODO move to subclass?
    def encode_poly(self, improvised_filepath, original_filepath, transpose_interval):
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
                'improvised_sustains': improvised_sustains + transpose_interval,
                'improvised_attacks': improvised_attacks,
                'original_sustains': original_sustains + transpose_interval,
                'original_attacks': original_attacks,
                'chord_name': chord_name
            })

            return pd.DataFrame(rows)

    # TODO choice for which extensions to add is a bit random
    def extended_chord_encoding(self, chord_pitches, chord_notes_count):
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
            upper_extension = chord_pitches[upper_extension_index] + self.OCTAVE_SEMITONES
            chord_pitches.append(upper_extension)

    def create_padded_tensors(self, sequence_size):
        padded_tensors = []

        loop_start = 0 - (sequence_size - 1)
        loop_end = self.encoded.shape[0]

        for offset_start in np.arange(loop_start, loop_end):
            start_idx = offset_start
            end_idx = offset_start + sequence_size

            padded_tensor = self.create_padded_tensor(start_idx, end_idx)
            padded_tensors.append(padded_tensor[None, :, :].int())

        return padded_tensors

    def create_padded_tensor(self, start_idx, end_idx):
        assert start_idx < end_idx
        assert end_idx > 0

        length = self.encoded.shape[0]

        padded_improvised_pitches = []
        padded_improvised_attacks = []

        offsets = torch.from_numpy(
            np.array([np.arange(start_idx, end_idx) % 48])
        ).long().clone()

        # TODO providing original melody on padding - might want to change it later if it affects learning

        original_pitches = torch.from_numpy(
            np.array([self.encoded.iloc[np.arange(start_idx, end_idx) % length]['original_pitch'].fillna(128)])
        ).long().clone()

        original_attacks = torch.from_numpy(
            np.array([self.encoded.iloc[np.arange(start_idx, end_idx) % length]['original_attack']])
        ).long().clone()

        # TODO using extended chord encoding - try other kinds later
        chord_encoding = self.extended_chord_encoding

        chord_pitches = torch.from_numpy(
            np.stack(
                self.encoded.iloc[np.arange(start_idx, end_idx) % length]['chord_name']
                    .apply(Chord) \
                    .apply(lambda x: x.getMIDI()) \
                    .apply(chord_encoding, chord_notes_count=7)
            )).long().clone().transpose(0, 1)

        # TODO check which symbols to use for padding

        # Left padding
        if start_idx < 0:
            left_improvised_pitches = torch.from_numpy(
                np.array([self.START_SYMBOL])
            ).long().clone().repeat(-start_idx, 1).transpose(0, 1)

            left_improvised_attacks = torch.from_numpy(
                np.array([0])
            ).long().clone().repeat(-start_idx, 1).transpose(0, 1)

            padded_improvised_pitches.append(left_improvised_pitches)
            padded_improvised_attacks.append(left_improvised_attacks)

        # Center
        slice_start = start_idx if start_idx > 0 else 0
        slice_end = end_idx if end_idx < length else length

        sliced_data = self.encoded.iloc[slice_start:slice_end]

        center_improvised_pitches = torch.from_numpy(
            np.array(sliced_data[['improvised_pitch']].fillna(128))
        ).long().clone().transpose(0, 1)

        center_improvised_attacks = torch.from_numpy(
            np.array(sliced_data[['improvised_attack']])
        ).long().clone().transpose(0, 1)

        padded_improvised_pitches.append(center_improvised_pitches)
        padded_improvised_attacks.append(center_improvised_attacks)

        # Right padding
        if end_idx > length:
            right_improvised_pitches = torch.from_numpy(
                np.array([self.END_SYMBOL])
            ).long().clone().repeat(end_idx - length, 1).transpose(0, 1)

            right_improvised_attacks = torch.from_numpy(
                np.array([0])
            ).long().clone().repeat(end_idx - length, 1).transpose(0, 1)

            padded_improvised_pitches.append(right_improvised_pitches)
            padded_improvised_attacks.append(right_improvised_attacks)

        improvised_pitches = torch.cat(padded_improvised_pitches, 1)
        improvised_attacks = torch.cat(padded_improvised_attacks, 1)

        return torch.cat([
            offsets,
            improvised_pitches,
            improvised_attacks,
            original_pitches,
            original_attacks,
            chord_pitches
        ], 0).transpose(0, 1)  # Do transpose?

    def to_midi(
            self,
            chord_progression: dict,
            out_file: str,
            out_bpm: int = 120,
    ):
        melody_instrument_name = "Tenor Sax"
        chord_instrument_name = "Acoustic Grand Piano"

        p = pm.PrettyMIDI()
        ts = pm.TimeSignature(
            self.time_signature[0],
            self.time_signature[1],
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

        attack_df = self.encoded[self.encoded['improvised_attack'] is True]
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
