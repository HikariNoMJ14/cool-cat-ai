import numpy as np
import pandas as pd
from mingus.core.notes import int_to_note

from ezchord import Chord
from melody import Melody
from poly_dataset import PolyDataset


class PolyTimestepDataset(PolyDataset):

    def encode_file(self, improvised_filepath):
        melody = Melody(improvised_filepath, self.version, self.base_folder)
        song_name = melody.song_name
        chord_progression = self.chord_progressions[song_name]

        improvised = pd.read_csv(improvised_filepath, index_col=0)
        improvised['end_ticks'] = improvised['ticks'] + improvised['duration']

        original_filepath = self.get_original_filepath(song_name)
        original = pd.read_csv(original_filepath, index_col=0)
        original['end_ticks'] = original['ticks'] + original['duration']

        beats_per_measure = chord_progression['time_signature'][0]
        flat_chord_progression = self.flatten_chord_progression(chord_progression)
        num_chord_progression_measures = int(len(flat_chord_progression) / beats_per_measure)

        n_ticks = self.ticks_per_measure * num_chord_progression_measures

        rows = []
        for i in range(n_ticks):
            offset = i % self.ticks_per_measure

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

            chord_name = flat_chord_progression[(i // beats_per_measure) % num_chord_progression_measures]
            chord_notes = "-".join([int_to_note(x) for x in np.array(Chord(chord_name).getMIDI()) % 12])

            rows.append({
                'offset': offset,
                'improvised_sustains': improvised_sustains,
                'improvised_attacks': improvised_attacks,
                'original_sustains': original_sustains,
                'original_attacks': original_attacks,
                'chord_notes': chord_notes
            })

        return pd.DataFrame(rows)
