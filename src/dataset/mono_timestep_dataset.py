import numpy as np
import pandas as pd
from mingus.core.notes import int_to_note

from ezchord import Chord
from melody import Melody
from src.dataset.mono_dataset import MonoDataset


class MonoTimestepDataset(MonoDataset):

    def encode_file(self, improvised_filepath, transpose_interval=0):
        melody = Melody(improvised_filepath, self.version, self.base_folder)
        song_name = melody.song_name

        try:
            chord_progression = self.chord_progressions[song_name]
            melody.set_song_structure(chord_progression)
        except KeyError:
            raise Exception(f'No chord progression for {song_name}')

        improvised = pd.read_csv(improvised_filepath, index_col=0)
        improvised['end_ticks'] = improvised['ticks'] + improvised['duration']

        original_filepath = self.get_original_filepath(song_name)
        original = pd.read_csv(original_filepath, index_col=0)
        original['end_ticks'] = original['ticks'] + original['duration']

        beats_per_measure = chord_progression['time_signature'][0]
        flat_chord_progression = self.flatten_chord_progression(chord_progression)
        num_chord_progression_measures = int(len(flat_chord_progression) / beats_per_measure)

        n_ticks = self.ticks_per_measure * num_chord_progression_measures

        if self.is_strongly_polyphonic(improvised):
            improvised = self.remove_strong_polyphony(improvised)

        if self.is_weakly_polyphonic(improvised):
            improvised = self.remove_weak_polyphony(improvised)

        if self.is_strongly_polyphonic(original):
            original = self.remove_strong_polyphony(original)

        if self.is_weakly_polyphonic(original):
            original = self.remove_weak_polyphony(original)

        rows = []
        for i in range(n_ticks):
            offset = i % self.ticks_per_measure

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

            chord_name = flat_chord_progression[(i // beats_per_measure) % num_chord_progression_measures]
            chord_notes = "-".join(
                [int_to_note(x) for x in np.array(Chord(chord_name).getMIDI() + transpose_interval) % 12])

            rows.append({
                'offset': offset,
                'improvised_pitch': improvised_pitch + transpose_interval,
                'improvised_attack': improvised_attack,
                'original_pitch': original_pitch + transpose_interval,
                'original_attack': original_attack,
                'chord_notes': chord_notes
            })

        encoded_df = pd.DataFrame(rows)

        self.normalize_pitch_range(encoded_df)
