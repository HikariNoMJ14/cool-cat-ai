import os
from glob import glob
from datetime import datetime
import torch
import numpy as np
from torch.utils.data import ConcatDataset

from src.melody import TimeStepMelody, DurationMelody
from src.utils import get_chord_progressions, get_filepaths, get_original_filepath
from src.utils.constants import OCTAVE_SEMITONES

dir_path = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.join(dir_path, '..', '..')


class MelodyDataset:
    VERSION = '1.2'

    def __init__(self,
                 encoding_type,
                 polyphonic,
                 sequence_size,
                 chord_encoding_type,
                 chord_extension_count,
                 duration_correction,
                 transpose_mode,
                 logger):
        super(MelodyDataset, self).__init__()

        self.dataset = None

        assert transpose_mode == 'all' or transpose_mode == 'c' or transpose_mode == 'none'

        self.encoding_type = encoding_type
        self.polyphonic = polyphonic
        self.sequence_size = sequence_size
        self.transpose_mode = transpose_mode
        self.chord_encoding_type = chord_encoding_type
        self.chord_extension_count = chord_extension_count
        self.duration_correction = duration_correction

        self.melody_info = {}
        self.chord_progressions = get_chord_progressions(src_path)

        if self.encoding_type == 'timestep':
            self.melody_class = TimeStepMelody
        elif self.encoding_type == 'duration':
            self.melody_class = DurationMelody
        else:
            raise Exception('Encoding type not supported!')

        self.out_folder = os.path.join(
            src_path,
            'data',
            'tensor_dataset',
            self.encoding_type,
            'poly' if self.polyphonic else 'mono'
        )
        self.name = f'sequence_{self.sequence_size}_' \
                    f'transpose_{self.transpose_mode}_' \
                    f'chord_{self.chord_encoding_type}_{self.chord_extension_count}'

        self.improvised_filepaths = get_filepaths('improvised')

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

    def split(self, split=(0.85, 0.10, 0.05), seed=None):
        assert sum(split) == 1

        dataset = self.dataset
        num_examples = len(dataset)

        return torch.utils.data.random_split(
            dataset, [
                int(round(num_examples * split[0])),
                int(round(num_examples * split[1])),
                int(round(num_examples * split[2]))
            ],
            generator=torch.Generator().manual_seed(seed)
        )

    # TODO also save song info to cross-reference songs and tensors
    def save(self):
        out_filename = f'{datetime.now().strftime("%Y_%m_%d_%H%M%S")}_' + self.name + '.pt'
        if not os.path.exists(self.out_folder):
            os.makedirs(self.out_folder)

        out_filepath = os.path.join(self.out_folder, out_filename)

        with open(out_filepath, 'wb+') as f:
            torch.save(self.dataset, f)

    def load(self, datetime=None):
        if datetime is None:
            # Get latest model
            candidate_filepaths = glob(os.path.join(self.out_folder, f'*{self.name}.pt'))
            load_filepath = sorted(candidate_filepaths)[-1]
        else:
            # Get model at specific datetime
            load_filepath = os.path.join(self.out_folder, f'{datetime}_{self.name}.pt')
            if not os.path.exists(load_filepath):
                raise FileNotFoundError(f'File {load_filepath} doesn\'t exist')

        with open(load_filepath, 'rb') as f:
            self.dataset = torch.load(f)

    def create(self):
        datasets = []

        if self.transpose_mode == 'c':
            transpose_interval = None  # TODO find transpose interval to C
            transpose_intervals = [transpose_interval]
        elif self.transpose_mode == 'all':
            transpose_intervals = np.arange(-6, 6)
        else:
            transpose_intervals = [0]

        for improvised_filepath in self.improvised_filepaths:
            self.logger.info(improvised_filepath)

            time_step_melody = self.melody_class(
                filepath=improvised_filepath,
                polyphonic=False,
                duration_correction=self.duration_correction
            )
            time_step_melody.set_song_structure(self.chord_progressions[time_step_melody.song_name])

            original_filepath = get_original_filepath(time_step_melody.song_name)

            time_step_melody.encode(improvised_filepath, original_filepath)
            time_step_melody.save_encoded()

            for transpose_interval in transpose_intervals:
                if len(transpose_intervals) > 1:
                    self.logger.debug(f'Transpose Interval: {transpose_interval}')

                melody_tensor = time_step_melody.to_tensor(transpose_interval)

                datasets.append(melody_tensor)

        self.dataset = ConcatDataset(datasets=datasets)

        self.save()
