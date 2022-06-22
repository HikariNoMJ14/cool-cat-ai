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
from src.utils.constants import REST_PITCH_SYMBOL, REST_ATTACK_SYMBOL, OCTAVE_SEMITONES, TICKS_PER_MEASURE

dir_path = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.join(dir_path, '..', '..')


class TimeStepMelody(Melody):
    VERSION = '1.2'

    def __init__(self, filepath, polyphonic=False,
                 chord_encoding_type='extended', chord_extension_count=7, duration_correction=0):
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
            original_attack = np.nan

            original_pitches = original[
                (original['ticks'] <= i) & (i < original['end_ticks'])
            ]

            if len(original_pitches) > 0:
                original_pitch = original_pitches['pitch'].values[0]

                if len(original_pitches) > 1:
                    print(original_pitches)
                    raise Exception('Error!!! not mono pitch on original')

            if not np.isnan(original_pitch):
                original_attacks = original[original['ticks'] == i]

                if len(original_attacks) > 0:
                    original_attack = 1

                    if len(original_attacks) > 1:
                        raise Exception('Error!!! not mono attack on original')
                else:
                    original_attack = 0

            improvised_pitch = np.nan
            improvised_attack = np.nan

            if improvised_filepath is not None:

                improvised_pitches = improvised[
                    (improvised['ticks'] <= i) & (i < improvised['end_ticks'])
                    ]

                if len(improvised_pitches) > 0:
                    improvised_pitch = improvised_pitches['pitch'].values[0]

                    if len(improvised_pitches) > 1:
                        raise Exception('Error!!! not mono pitch on improvised')

                if not np.isnan(improvised_pitch):
                    improvised_attacks = improvised[improvised['ticks'] == i]

                    if len(improvised_attacks) > 0:
                        improvised_attack = 1

                        if len(improvised_attacks) > 1:
                            raise Exception('Error!!! not mono attack on improvised')
                    else:
                        improvised_attack = 0

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

    def to_tensor(self, transpose_interval, metadata):
        offsets = torch.from_numpy(
            np.array([self.encoded['offset']])
        ).long().clone()

        improvised_pitches = torch.from_numpy(
            np.array([(self.encoded['improvised_pitch'] + transpose_interval).fillna(REST_PITCH_SYMBOL)])
        ).long().clone()

        improvised_attacks = torch.from_numpy(
            np.array([self.encoded['improvised_attack'].fillna(REST_ATTACK_SYMBOL)])
        ).long().clone()

        original_pitches = torch.from_numpy(
            np.array([(self.encoded['original_pitch'] + transpose_interval).fillna(REST_ATTACK_SYMBOL)])
        ).long().clone()

        original_attacks = torch.from_numpy(
            np.array([self.encoded['original_attack'].fillna(REST_ATTACK_SYMBOL)])
        ).long().clone()

        metadata = torch.from_numpy(
            np.tile(metadata, (self.encoded.shape[0], 1))
        ).long().clone().transpose(0, 1)

        chord_pitches = torch.from_numpy(
            np.stack(
                self.encoded['chord_name'].apply(
                    lambda x: self.transpose_chord(x, transpose_interval)
                ).fillna(REST_PITCH_SYMBOL)
            )
        ).long().clone().transpose(0, 1)

        return torch.cat([
            offsets,
            improvised_pitches,
            improvised_attacks,
            original_pitches,
            original_attacks,
            metadata,
            chord_pitches
        ], 0).transpose(0, 1)[None, :, :]

    def to_midi(
            self,
            out_filepath: str,
            out_bpm: int = 120,
    ):
        melody_instrument_name = "Vibraphone"
        chord_instrument_name = "Acoustic Grand Piano"
        drum_instrument_name = "Acoustic Grand Piano"

        p = pm.PrettyMIDI()
        ts = pm.TimeSignature(
            self.chord_progression_time_signature[0],
            self.chord_progression_time_signature[1],
            0
        )

        p.time_signature_changes.append(ts)

        max_measure = (self.encoded['offset'] // TICKS_PER_MEASURE).max()

        # Add melody
        melody = pm.Instrument(
            program=pm.instrument_name_to_program(melody_instrument_name),
            name="melody"
        )

        melody_multiplier = (240 / out_bpm) / (self.FINAL_TICKS_PER_BEAT * self.chord_progression_time_signature[0])

        self.encoded['ticks'] = self.encoded.index

        attack_df = self.encoded[self.encoded['improvised_attack'] == True]
        attack_df.reset_index(inplace=True)

        for i, row in attack_df.iterrows():
            start = row.ticks * melody_multiplier

            duration = 0

            next_idx = min(len(attack_df) - 1, i + 1)
            next_att = attack_df.iloc[next_idx]['ticks']

            for j, row2 in self.encoded.iloc[row.ticks:next_att].iterrows():
                if row2['improvised_pitch'] == row['improvised_pitch']:
                    duration += 1

            end = start + duration * melody_multiplier

            note = pm.Note(
                velocity=127,
                pitch=int(row["improvised_pitch"]),
                start=start,
                end=end,
            )
            melody.notes.append(note)

        p.instruments.append(melody)

        # Add chords
        start = 0
        beat_n = 0
        previous_chord_name = ''

        chords = pm.Instrument(
            program=pm.instrument_name_to_program(chord_instrument_name),
            name="chords"
        )

        chord_multiplier = 60 / out_bpm

        # TODO Generalize for melodies longer than one cycle
        for section in self.song_structure['sections']:
            for chord_name in self.song_structure['progression'][section]:
                chord_notes = Chord(chord_name).getMIDI()

                # Use the tonic on the first beat, the fifth on the third beat
                # and the full chord (minus the fifth) on beats 2 and 4
                # TODO only works with 4/4
                if beat_n == 0:
                    note = pm.Note(
                        velocity=64,
                        pitch=int(chord_notes[0]),
                        start=start,
                        end=start + 0.25,
                    )
                    chords.notes.append(note)
                elif beat_n == 2:
                    note = pm.Note(
                        velocity=64,
                        pitch=int(chord_notes[3]) - OCTAVE_SEMITONES * 2,
                        start=start,
                        end=start + (chord_multiplier / 2),
                    )
                    chords.notes.append(note)
                else:
                    for extension, chord_note in enumerate(chord_notes[1:]):
                        if extension != 2:  # don't add the fifth
                            note = pm.Note(
                                velocity=64,
                                pitch=int(chord_note),
                                start=start,
                                end=start + (chord_multiplier / 2),
                            )
                            chords.notes.append(note)

                if beat_n == 0 or chord_name != previous_chord_name:
                    # Add chord annotation
                    chord_annotation = pm.Lyric(chord_name, start)

                    p.lyrics.append(chord_annotation)
                    previous_chord_name = chord_name

                start += chord_multiplier
                beat_n = (beat_n + 1) % 4

        p.instruments.append(chords)

        # Add drums
        start = 0
        beat_n = 0

        drums = pm.Instrument(
            program=pm.instrument_name_to_program(drum_instrument_name),
            is_drum=True,
            name="drums"
        )

        drums_multiplier = 60 / out_bpm

        measure = 0
        while measure <= max_measure:
            # Add on-beat ride cymbal
            note = pm.Note(
                velocity=72,
                pitch=pm.drum_name_to_note_number("Ride Cymbal 1"),
                start=start,
                end=start + (drums_multiplier / 2),
            )
            drums.notes.append(note)

            # Add 'swung' ride cymbal
            if beat_n == 0 or beat_n == 2:
                note = pm.Note(
                    velocity=72,
                    pitch=pm.drum_name_to_note_number("Ride Cymbal 1"),
                    start=start - 0.1333,
                    end=start + (drums_multiplier / 2) - 0.1333,
                )
                drums.notes.append(note)

            # Add bass drum
            if beat_n == 0:
                note = pm.Note(
                    velocity=64,
                    pitch=pm.drum_name_to_note_number("Acoustic Bass Drum"),
                    start=start,
                    end=start + (drums_multiplier / 2),
                )
                drums.notes.append(note)

                # Add crash cymbal
                if measure % 8 == 0 and measure != 0:
                    note = pm.Note(
                        velocity=72,
                        pitch=pm.drum_name_to_note_number("Crash Cymbal 1"),
                        start=start,
                        end=start + (drums_multiplier / 2),
                    )
                    drums.notes.append(note)

            # Add hi-hat pedal
            if beat_n == 1 or beat_n == 3:
                note = pm.Note(
                    velocity=96,
                    pitch=pm.drum_name_to_note_number("Pedal Hi-hat"),
                    start=start,
                    end=start + (drums_multiplier / 2),
                )
                drums.notes.append(note)

            measure += 1 / 4
            start += drums_multiplier
            beat_n = (beat_n + 1) % 4

        p.instruments.append(drums)

        p.write(out_filepath)
