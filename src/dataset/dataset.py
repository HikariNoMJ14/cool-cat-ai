import os
from abc import ABC, abstractmethod
from glob import glob

import numpy as np

from src.utils import get_chord_progressions, flatten_chord_progression
# from src.dataset.example import Example


class Dataset(ABC):
    octave_semitones = 12
    ticks_per_measure = 48

    version = '1.2'

    sources = {
        'original': [
            'Real Book'
        ],
        'improvised': [
            'Doug McKenzie',
            'Jazz-Midi',
            'Jazz Standards',
            'JazzPage',
            'MidKar',
            'Oocities',
            'Weimar DB'
        ]
    }

    def __init__(self, batch_size, sequence_size,
                 chord_encoding_type, chord_extension_count,
                 transpose_mode,
                 base_folder='.'):
        super(Dataset, self).__init__()

        assert transpose_mode == 'all' or transpose_mode == 'c' or transpose_mode == 'none'

        self.transpose_mode = transpose_mode

        self.batch_size = batch_size
        self.sequence_size = sequence_size
        self.chord_encoding_type = chord_encoding_type
        self.chord_extension_count = chord_extension_count

        self.melody_info = {}
        self.chord_progressions = get_chord_progressions(base_folder)

        self.input_data_folder = os.path.join(base_folder, f'data/split_melody_data/{self.version}')
        self.improvised_filepaths = self.get_filepaths('improvised')
        self.original_filepaths = self.get_filepaths('original')

    def get_filepaths(self, mode):
        filepaths = []
        for source in self.sources[mode]:
            filepaths += [y for x in os.walk(os.path.join(self.input_data_folder, source))
                          for y in glob(os.path.join(x[0], '*.csv'))]
        return filepaths

    def get_original_filepath(self, song_name):
        original_filepaths = [filepath for filepath in self.original_filepaths if song_name in filepath]

        if len(original_filepaths) > 1:
            raise Exception(f'Multiple original files match the song name {song_name}')

        return original_filepaths[0]

    @staticmethod
    def multiple_pitches_to_string(pitches):
        return "-".join(str(x) for x in pitches)

    @staticmethod
    def is_valid_min_pitch(pitches):
        return pitches.min() >= 0

    @staticmethod
    def is_valid_max_pitch(pitches):
        return pitches.max() <= 127

    def is_valid_pitch_range(self, pitches):
        return self.is_valid_min_pitch(pitches) and self.is_valid_max_pitch(pitches)

    def transpose_pitches(self, pitches, interval):
        new_pitches = pitches.copy() + interval

        c = 0
        while True:
            if self.is_valid_pitch_range(new_pitches):
                return new_pitches
            elif not self.is_valid_max_pitch(new_pitches):
                new_pitches -= self.octave_semitones
            elif not self.is_valid_min_pitch(new_pitches):
                new_pitches += self.octave_semitones

            if c > 20:
                raise Exception(f'Valid pitch range can\'t be found')

            c += 1

    @abstractmethod
    def encode_file(self, improvised_filepath):
        pass

    def create(self):
        for improvised_filepath in self.improvised_filepaths:
            df = self.encode_file(improvised_filepath)

            if self.transpose_mode == 'c':
                transpose_interval = None  # TODO find transpose interval to C
                transpose_intervals = [transpose_interval]
            elif self.transpose_mode == 'all':
                transpose_intervals = np.arange(-6, 6)
            else:
                transpose_intervals = [0]

            for interval in transpose_intervals:
                transposed_df = df.copy()
                transposed_df.loc[:, 'improvised_pitch'] = self.transpose_pitches(
                    transposed_df['improvised_pitch'], interval)
                transposed_df.loc[:, 'original_pitch'] = self.transpose_pitches(
                    transposed_df['original_pitch'], interval)

                for i, row in transposed_df.iterrows():
                    pass


class Example:
    past_features = None
    present_features = None
    future_features = None

    target = None


if __name__ == "__main__":
    import numpy as np
    from src.dataset.mono_timestep_dataset import MonoTimestepDataset

    a = np.array([129, 80, 34, 20])
    d = MonoTimestepDataset(120, 100, 'compressed', None, 'all', '../..')

    print(d.transpose_pitches(a, 0))
