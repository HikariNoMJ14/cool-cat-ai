import os
from glob import glob
from datetime import datetime
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from src.melody import TimeStepMelody
from src.utils import get_chord_progressions, filepath_to_song_name

dir_path = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.join(dir_path, '..', '..')


class CustomTensorDataset(TensorDataset):
    def __init__(self, root):
        super(CustomTensorDataset, self).__init__()

        self.root = root
        self.files = os.listdir(root)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample = torch.load(os.path.join(self.root, self.files[idx]))  # load the features of this sample

        return sample


class SplitDataset:
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
        super(SplitDataset, self).__init__()

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

        self.name = f'sequence_{self.sequence_size}_' \
                    f'transpose_{self.transpose_mode}_' \
                    f'chord_{self.chord_encoding_type}_{self.chord_extension_count}'

        self.out_folder = os.path.join(
            src_path,
            'data',
            'tensor_dataset',
            self.encoding_type,
            'poly' if self.polyphonic else 'mono',
            self.name
        )

        self.improvised_filepaths = self.get_filepaths('improvised')
        self.original_filepaths = self.get_filepaths('original')

    def __len__(self):
        return len(self.tensor_dataset)

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

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
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

    def load(self, datetime=None):
        if datetime is None:
            # Get latest model
            candidate_folders = glob(os.path.join(self.out_folder, "*"))
            if len(candidate_folders) == 0:
                raise FileNotFoundError(f'No folders under {candidate_folders}')
            load_folder = sorted(candidate_folders)[-1]
        else:
            # Get model at specific datetime
            load_folder = os.path.join(self.out_folder, f'{datetime}')
            if not os.path.exists(load_folder):
                raise FileNotFoundError(f'Folder {load_folder} doesn\'t exist')

        self.tensor_dataset = CustomTensorDataset(load_folder)

    def create(self):
        out_filepath = os.path.join(self.out_folder, f'{datetime.now().strftime("%Y_%m_%d_%H%M%S")}')

        if not os.path.exists(out_filepath):
            os.makedirs(out_filepath)

        for improvised_filepath in self.improvised_filepaths:
            print(improvised_filepath)
            if self.transpose_mode == 'c':
                transpose_interval = None  # TODO find transpose interval to C
                transpose_intervals = [transpose_interval]
            elif self.transpose_mode == 'all':
                transpose_intervals = np.arange(-6, 6)
            else:
                transpose_intervals = [0]

            ts_melody = TimeStepMelody(improvised_filepath, polyphonic=False)
            ts_melody.set_song_structure(self.chord_progressions[ts_melody.song_name])

            original_filepath = self.get_original_filepath(ts_melody.song_name)

            ts_melody.encode(improvised_filepath, original_filepath)

            for transpose_interval in transpose_intervals:
                if len(transpose_intervals) > 1:
                    print(f'Transpose Interval: {transpose_interval}')

                loop_start = -self.sequence_size
                loop_end = ts_melody.encoded.shape[0]

                for offset_start in np.arange(loop_start, loop_end):
                    start_idx = offset_start
                    end_idx = offset_start + self.sequence_size

                    padded_tensor = ts_melody.create_padded_tensor(
                        start_idx,
                        end_idx,
                        transpose_interval
                    )[None, :, :].int()

                    out_filename = os.path.join(
                        out_filepath,
                        f'{ts_melody.source}#'
                        f'{ts_melody.filename.replace(".csv", "")}#'
                        f'{transpose_interval}#'
                        f'{offset_start}.pt'
                    )

                    with open(out_filename, 'wb+') as fp:
                        torch.save(padded_tensor, fp)
