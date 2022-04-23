import os
from glob import glob
import pickle
from datetime import datetime
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from src.melody import TimeStepMelody
from src.utils import get_chord_progressions, filepath_to_song_name

dir_path = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.join(dir_path, '..', '..')


class Dataset:
    VERSION = '1.2'

    OCTAVE_SEMITONES = 12

    SOURCES = {
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

    def __init__(self,
                 encoding_type,
                 polyphonic,
                 sequence_size,
                 chord_encoding_type,
                 chord_extension_count,
                 transpose_mode):
        super(Dataset, self).__init__()

        self.tensor_dataset = None

        assert transpose_mode == 'all' or transpose_mode == 'c' or transpose_mode == 'none'

        self.encoding_type = encoding_type
        self.polyphonic = polyphonic
        self.sequence_size = sequence_size
        self.transpose_mode = transpose_mode
        self.chord_encoding_type = chord_encoding_type
        self.chord_extension_count = chord_extension_count

        self.melody_info = {}
        self.chord_progressions = get_chord_progressions(src_path)

        self.input_data_folder = os.path.join(
            src_path,
            'data',
            'finalised',
            'csv',
        )

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

        self.improvised_filepaths = self.get_filepaths('improvised')
        self.original_filepaths = self.get_filepaths('original')

    def get_filepaths(self, mode):
        filepaths = []
        for source in self.SOURCES[mode]:
            filepaths += [y for x in os.walk(os.path.join(self.input_data_folder, source))
                          for y in glob(os.path.join(x[0], '*.csv'))]
        return filepaths

    def get_original_filepath(self, song_name):
        original_filepaths = [filepath for filepath in self.original_filepaths
                              if filepath_to_song_name(filepath) == song_name]

        if len(original_filepaths) > 1:
            raise Exception(f'Multiple original files match the song name {song_name}, {original_filepaths}')

        return original_filepaths[0]

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
                new_pitches -= self.OCTAVE_SEMITONES
            elif not self.is_valid_min_pitch(new_pitches):
                new_pitches += self.OCTAVE_SEMITONES

            if c > 20:
                raise Exception(f'Valid pitch range can\'t be found')

            c += 1

    def data_loaders(self, batch_size, split=(0.85, 0.10, 0.05), seed=None):

        assert sum(split) == 1

        dataset = self.tensor_dataset
        num_examples = len(dataset)
        train_dataset, \
        val_dataset, \
        eval_dataset = torch.utils.data.random_split(
            dataset, [
                int(round(num_examples * split[0])),
                int(round(num_examples * split[1])),
                int(round(num_examples * split[2]))
            ],
            generator=torch.Generator().manual_seed(seed)
        )

        # a, b, _ = split
        # train_dataset = TensorDataset(*dataset[:int(a * num_examples)])
        # val_dataset = TensorDataset(*dataset[int(a * num_examples):int((a + b) * num_examples)])
        # eval_dataset = TensorDataset(*dataset[int((a + b) * num_examples):])

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,  # TODO change to 4
            pin_memory=True,
            drop_last=True,
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=False,
            drop_last=True,
        )

        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=False,
            drop_last=True,
        )

        return train_dataloader, val_dataloader, eval_dataloader

    # TODO also save song info to cross-reference songs and tensors
    def save(self):
        out_filename = f'{datetime.now().strftime("%Y_%m_%d_%H%M%S")}_' + self.name + '.pickle'
        if not os.path.exists(self.out_folder):
            os.makedirs(self.out_folder)

        out_filepath = os.path.join(self.out_folder, out_filename)

        with open(out_filepath, 'wb+') as f:
            pickle.dump(self.tensor_dataset, f)

    def load(self, datetime=None):
        if datetime is None:
            # Get latest model
            candidate_filepaths = glob(os.path.join(self.out_folder, f'*{self.name}.pickle'))
            load_filepath = sorted(candidate_filepaths)[-1]
        else:
            # Get model at specific datetime
            load_filepath = os.path.join(self.out_folder, f'{datetime}_{self.name}.pickle')
            if not os.path.exists(load_filepath):
                raise FileNotFoundError(f'File {load_filepath} doesn\'t exist')

        with open(load_filepath, 'rb') as f:
            self.tensor_dataset = pickle.load(f)

    def create(self):
        dataset = []

        for improvised_filepath in self.improvised_filepaths:
            print(improvised_filepath)
            if self.transpose_mode == 'c':
                transpose_interval = None  # TODO find transpose interval to C
                transpose_intervals = [transpose_interval]
            elif self.transpose_mode == 'all':
                transpose_intervals = np.arange(-6, 6)
            else:
                transpose_intervals = [0]

            for interval in transpose_intervals:
                if len(transpose_intervals) > 1:
                    print(f'Transpose Interval: {interval}')

                time_step_melody = TimeStepMelody(improvised_filepath, polyphonic=False)
                time_step_melody.set_song_structure(self.chord_progressions[time_step_melody.song_name])

                original_filepath = self.get_original_filepath(time_step_melody.song_name)

                time_step_melody.encode(improvised_filepath, original_filepath, transpose_interval=interval)

                dataset += time_step_melody.create_padded_tensors(self.sequence_size)

        self.tensor_dataset = TensorDataset(torch.cat(dataset, 0))

        self.save()


if __name__ == "__main__":
    print(torch.cuda.is_available())
    d = Dataset(sequence_size=48 * 2,
                encoding_type='timestep',
                polyphonic=False,
                chord_encoding_type='extended',
                chord_extension_count=7,
                transpose_mode='none')

    d.create()

    print(d.tensor_dataset)
