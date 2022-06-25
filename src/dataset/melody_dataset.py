import os
import re
from glob import glob
from datetime import datetime

import music21
import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import ConcatDataset

from src.melody import TimeStepMelody, DurationMelody
from src.utils import get_chord_progressions, get_filepaths, get_original_filepath
from src.utils.constants import OCTAVE_SEMITONES

dir_path = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.join(dir_path, '..', '..')


class MelodyDataset:
    VERSION = '1.2'
    METADATA_COUNT = 1

    def __init__(self,
                 encoding_type,
                 polyphonic,
                 chord_encoding_type,
                 chord_extension_count,
                 transpose_mode,
                 logger):
        super(MelodyDataset, self).__init__()

        self.tensor_dataset = None
        self.melody_dataset = []

        assert transpose_mode == 'all' or transpose_mode == 'c' or transpose_mode == 'none'

        self.encoding_type = encoding_type
        self.polyphonic = polyphonic
        self.transpose_mode = transpose_mode
        self.chord_encoding_type = chord_encoding_type
        self.chord_extension_count = chord_extension_count

        self.melody_info = {}
        self.chord_progressions = get_chord_progressions(src_path)

        if self.encoding_type == 'timestep' \
                or self.encoding_type == 'timestep_base' \
                or self.encoding_type == 'timestep_chord':
            self.melody_class = TimeStepMelody
        elif self.encoding_type == 'duration' \
                or self.encoding_type == 'duration_base' \
                or self.encoding_type == 'duration_chord':
            self.melody_class = DurationMelody
        else:
            raise Exception('Encoding type not supported!')

        self.out_folder = os.path.join(
            src_path,
            'data',
            'tensor_dataset',
            self.encoding_type.split('_')[0],
            'poly' if self.polyphonic else 'mono'
        )
        self.name = f'transpose_{self.transpose_mode}_' \
                    f'chord_{self.chord_encoding_type}_{self.chord_extension_count}'

        self.improvised_filepaths = get_filepaths('improvised')

        self.metadata = pd.read_csv(os.path.join(src_path, 'data', 'finalised', 'metadata.csv'), index_col=0)

        self.logger = logger

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
                new_pitches -= OCTAVE_SEMITONES
            elif not self.is_valid_min_pitch(new_pitches):
                new_pitches += OCTAVE_SEMITONES

            if c > 20:
                raise Exception(f'Valid pitch range can\'t be found')

            c += 1

    def find_interval_to_c(self, melody):
        transpose_semitones = music21.interval.Interval(
            music21.pitch.Pitch(melody.chord_progression_key),
            music21.pitch.Pitch('C')
        ).semitones

        transpose_semitones = ((transpose_semitones + 6) % 12) - 6

        return transpose_semitones

    def split(self, split=(0.85, 0.15), seed=None):
        assert sum(split) == 1

        tensor_dataset = self.tensor_dataset
        num_examples = len(tensor_dataset)

        return torch.utils.data.random_split(
            tensor_dataset, [
                int(round(num_examples * split[0])),
                int(round(num_examples * split[1]))
            ],
            generator=torch.Generator().manual_seed(seed)
        )

    def get_metadata(self, melody):

        metadata = self.metadata[
            (self.metadata['song_name'] == melody.song_name) &
            (self.metadata['source'] == melody.source) &
            (self.metadata['filename'] == re.sub(' -[0-9,o]*-', '', melody.filename))
            ]

        if len(metadata) == 1:
            return metadata.iloc[0, -self.METADATA_COUNT:].values.astype(int)
        else:
            print('Problem with metadata')

    def save(self):
        out_filename = f'{datetime.now().strftime("%Y_%m_%d_%H%M%S")}_' + self.name
        melody_out_filename = out_filename + '.pickle'
        tensor_out_filename = out_filename + '.pt'

        if not os.path.exists(self.out_folder):
            os.makedirs(self.out_folder)

        melody_out_filepath = os.path.join(self.out_folder, melody_out_filename)
        tensor_out_filepath = os.path.join(self.out_folder, tensor_out_filename)

        with open(melody_out_filepath, 'wb+') as f:
            pickle.dump(self.melody_dataset, f)

        with open(tensor_out_filepath, 'wb+') as f:
            torch.save(self.tensor_dataset, f)

    def load(self, datetime=None):
        if datetime is None:
            # Get the latest model
            candidate_filepaths = glob(os.path.join(self.out_folder, f'*{self.name}.pt'))
            load_filepath = sorted(candidate_filepaths)[-1]
        else:
            # Get model at specific datetime
            load_filepath = os.path.join(self.out_folder, f'{datetime}_{self.name}.pt')
            if not os.path.exists(load_filepath):
                raise FileNotFoundError(f'File {load_filepath} doesn\'t exist')

        with open(load_filepath, 'rb') as f:
            self.tensor_dataset = torch.load(f)

    def create(self):
        tensor_dataset = []

        for improvised_filepath in self.improvised_filepaths:
            self.logger.info(improvised_filepath)

            melody = self.melody_class(
                filepath=improvised_filepath,
                polyphonic=False,
                chord_encoding_type=self.chord_encoding_type,
                chord_extension_count=self.chord_extension_count
            )
            melody.set_song_structure(self.chord_progressions[melody.song_name])

            original_filepath = get_original_filepath(melody.song_name)

            melody.encode(improvised_filepath, original_filepath)
            melody.save_encoded()

            metadata = self.get_metadata(melody)

            if self.transpose_mode == 'c':
                transpose_interval = self.find_interval_to_c(melody)
                transpose_intervals = [transpose_interval]
            elif self.transpose_mode == 'all':
                transpose_intervals = np.arange(-6, 6)
            else:
                transpose_intervals = [0]

            for transpose_interval in transpose_intervals:
                if len(transpose_intervals) > 1:
                    self.logger.debug(f'Transpose Interval: {transpose_interval}')

                melody_tensor = melody.to_tensor(
                    transpose_interval,
                    metadata
                )

                tensor_dataset.append(melody_tensor)

                self.melody_dataset.append(melody)

        self.tensor_dataset = ConcatDataset(datasets=tensor_dataset)

        self.save()
