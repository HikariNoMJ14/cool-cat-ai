import os

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from src.model import ChordModel
from src.model import DurationBaseModel
from src.generator import DurationChordGenerator
from src.utils.constants import TICKS_PER_MEASURE

dir_path = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.join(dir_path, '..', '..', '..')


class DurationChordModel(DurationBaseModel, ChordModel):
    # Input data semantics
    TENSOR_IDX_MAPPING = {
        'original_flag': 0,
        'ticks': 1,
        'offset': 2,
        'pitch': 3,
        'duration': 4,
        'metadata': 5,
        'chord_pitches_start': 6
    }

    FEATURES = {
        'past_improvised': [
            'offset',
            'pitch', 'duration'
        ],
        'present': [
            'offset',
            'metadata'
        ]
    }

    LABELS = [
        'pitch', 'duration'
    ]

    METRICS_LIST = [
        'pitch_loss', 'duration_loss',
        'pitch_top1', 'pitch_top3', 'pitch_top5',
        'duration_top1', 'duration_top3', 'duration_top5'
    ]

    PLOTTED_METRICS = ['loss', 'pitch_loss', 'duration_loss']

    GENERATOR_CLASS = DurationChordGenerator

    def __init__(self, dataset=None, logger=None, save_path=os.path.join(src_path, 'results'), **kwargs):
        super(DurationChordModel, self).__init__(dataset, logger, save_path, **kwargs)

        #  offset +
        #  pitch + duration +
        #  chord_pitch * number_of_pitches
        past_lstm_input_size = self.embedding_size + \
                               self.embedding_size + self.embedding_size + \
                               self.embedding_size * self.chord_extension_count

        self.logger.debug(f'Model past LSTM input size: {past_lstm_input_size}')

        self.past_improvised_lstm = nn.LSTM(
            input_size=past_lstm_input_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            dropout=self.lstm_dropout_rate,
            batch_first=True
        )

        #  offset +
        #  metadata +
        #  chord_pitch * number_of_pitches
        present_nn_input_size = self.embedding_size + \
                                self.embedding_size + \
                                self.embedding_size * self.chord_extension_count

        self.logger.debug(f'Model present LSTM input size: {present_nn_input_size}')

        self.present_nn = nn.Sequential(
            nn.Linear(present_nn_input_size, self.nn_hidden_size),
            nn.ReLU(),
            nn.Dropout(self.nn_dropout_rate),
            nn.Linear(self.nn_hidden_size, self.nn_output_size),
            nn.ReLU(),
            nn.Dropout(self.nn_dropout_rate)
        )

        merge_nn_input_size = self.lstm_hidden_size + self.nn_output_size
        merge_nn_output_size = self.embedding_size + self.duration_size

        self.merge_nn = nn.Sequential(
            nn.Linear(merge_nn_input_size, self.nn_hidden_size),
            nn.ReLU(),
            nn.Dropout(self.nn_dropout_rate),
            nn.Linear(self.nn_hidden_size, merge_nn_output_size)
        )

        self.duration_decoder = nn.Linear(self.embedding_size, self.duration_size)

        # Tie duration encoder and decoder weights
        self.duration_decoder.weight = self.duration_encoder[0].weight

    def prepare_past_lstm_input(self, past):
        # Extract features from past tensor
        past_offsets = self.extract_features(past, 'offset', 0)  # TODO fix
        past_pitches = self.extract_features(past, 'pitch', 1)
        past_durations = self.extract_features(past, 'duration', 2)
        past_durations = self.convert_durations_to_ids(past_durations)
        past_chord_pitches = self.extract_chords(past, (3, 10))

        assert past_offsets.max() <= self.offset_size
        assert past_pitches.max() <= self.pitch_size
        assert past_durations.max() <= self.duration_size
        assert past_chord_pitches.max() <= self.pitch_size

        # Encode past features
        past_offset_embedding = self.offset_encoder(past_offsets)
        past_pitch_embedding = self.pitch_encoder(past_pitches)
        past_duration_embedding = self.duration_encoder(past_durations)
        past_chord_pitches_embedding = self.encode_chord_pitches(past_chord_pitches)

        return torch.cat([
            past_offset_embedding,
            past_pitch_embedding, past_duration_embedding,
            past_chord_pitches_embedding
        ], 2)

    def prepare_present_nn_input(self, present):
        # Extract features from present tensor
        present_offsets = self.extract_features(present, 'offset', 0)  # TODO fix
        present_metadata = self.extract_features(present, 'metadata', 1)
        present_chord_pitches = self.extract_chords(present, (2, 9))

        # Encode present features
        present_offset_embedding = self.offset_encoder(present_offsets)
        present_metadata_embedding = self.metadata_encoder(present_metadata)

        present_chord_pitches_embedding = self.encode_chord_pitches(present_chord_pitches)

        return torch.cat([
            present_offset_embedding,
            present_metadata_embedding,
            present_chord_pitches_embedding
        ], 2)

    def forward(self, features):
        past_improvised = features[0]
        present = features[1]

        self.cuda()

        # Past Improvised LSTM
        past_improvised_lstm_input = self.prepare_past_lstm_input(past_improvised)
        past_improvised_lstm_hidden = self.init_hidden(batch_size=past_improvised.size(0))
        past_improvised_lstm_output, \
        past_improvised_lstm_hidden = self.past_improvised_lstm(
            past_improvised_lstm_input,
            past_improvised_lstm_hidden
        )
        past_improvised_lstm_output = past_improvised_lstm_output[:, -1, :]

        # Present NN
        present_nn_input = self.prepare_present_nn_input(present)
        present_nn_input = present_nn_input.view(present.size(0), -1)
        present_nn_output = self.present_nn(present_nn_input)

        # Merge NN
        merge_nn_input = torch.cat([
            past_improvised_lstm_output,
            present_nn_output
        ], 1)
        merge_nn_output = self.merge_nn(merge_nn_input)

        output_improvised_pitch = self.pitch_decoder(torch.sigmoid(merge_nn_output[:, :self.embedding_size]))
        output_improvised_duration = self.duration_decoder(torch.sigmoid(merge_nn_output[:, self.embedding_size:]))

        if self.normalize:
            output_improvised_pitch = F.normalize(output_improvised_pitch, p=2, dim=1)
            output_improvised_duration = F.normalize(output_improvised_duration, p=2, dim=1)

        return output_improvised_pitch, output_improvised_duration

    def prepare_examples(self, batch):
        batch_size, sequence_size, _ = batch.size()
        middle_tick = sequence_size // 2

        assert batch[:, middle_tick:middle_tick + 1, 1].eq(self.start_pitch_symbol).count_nonzero() == 0
        assert batch[:, middle_tick:middle_tick + 1, 1].eq(self.end_pitch_symbol).count_nonzero() == 0

        if self.start_pitch_symbol != self.start_pitch_symbol:
            assert batch[:, :middle_tick, 1].eq(self.end_pitch_symbol).count_nonzero() == 0
            assert batch[:, middle_tick + 1:, 1].eq(self.start_pitch_symbol).count_nonzero() == 0

        past_improvised_tensor_indices = [self.TENSOR_IDX_MAPPING[feature]
                                          for feature in self.FEATURES['past_improvised']]
        past_improvised_tensor_indices += self.chord_tensor_idx
        past_improvised_tensor_indices = [0, 1, 2] + list(range(4, 4 + self.chord_extension_count))  # TODO fix!!!
        past_improvised = self.mask_entry(
            batch[:, :middle_tick, :],
            past_improvised_tensor_indices,
            dim=2
        )

        # Remove improvised pitch and duration from present tick
        present_tensor_indices = [self.TENSOR_IDX_MAPPING[feature]
                                  for feature in self.FEATURES['present']]
        present_tensor_indices += self.chord_tensor_idx
        present_tensor_indices = [0] + [3] + list(range(4, 4 + self.chord_extension_count))  # TODO fix!!!
        present = self.mask_entry(
            batch[:, middle_tick:middle_tick + 1, :],
            present_tensor_indices,
            dim=2
        )

        # Remove everything but improvised pitch and duration to get label
        label_tensor_indices = [self.TENSOR_IDX_MAPPING[feature]
                                for feature in self.LABELS]
        label_tensor_indices = range(1, 3)  # TODO fix!!!
        label = self.mask_entry(
            batch[:, middle_tick:middle_tick + 1:, :],
            label_tensor_indices,
            dim=2
        )
        label = label.view(batch_size, -1)

        assert present[:, :, 1].eq(self.start_pitch_symbol).count_nonzero() == 0 and \
               present.eq(self.end_pitch_symbol).count_nonzero() == 0
        assert label[:, 0].eq(self.start_pitch_symbol).count_nonzero() == 0 and \
               label[:, 0].eq(self.end_pitch_symbol).count_nonzero() == 0

        if self.start_pitch_symbol != self.start_pitch_symbol:
            assert past_improvised[:, :, 1].eq(self.end_pitch_symbol).count_nonzero() == 0

        return (past_improvised, present), label

    def create_padded_tensor(self, example, index):
        start_idx = index
        end_idx = index + self.sequence_size

        metadata_start = self.TENSOR_IDX_MAPPING['metadata']
        metadata_end = self.TENSOR_IDX_MAPPING['metadata'] + self.METADATA_IDX_COUNT

        chord_pitches_start = self.TENSOR_IDX_MAPPING['chord_pitches_start']
        chord_pitches_end = self.TENSOR_IDX_MAPPING['chord_pitches_start'] + self.chord_extension_count

        length = example.size(0)

        padded_offsets = []
        padded_pitches = []
        padded_durations = []
        padded_metadata = []
        padded_chord_pitches = []

        slice_start = start_idx if start_idx > 0 else 0
        slice_end = end_idx if end_idx < length else length

        sliced_data = example[slice_start:slice_end]

        center_padded_offsets = sliced_data[None, :, self.TENSOR_IDX_MAPPING['offset']]
        center_padded_pitches = sliced_data[None, :, self.TENSOR_IDX_MAPPING['pitch']]
        center_padded_durations = sliced_data[None, :, self.TENSOR_IDX_MAPPING['duration']]
        center_padded_metadata = sliced_data[:, metadata_start:metadata_end].transpose(0, 1)
        center_padded_chord_pitches = sliced_data[:, chord_pitches_start:chord_pitches_end].transpose(0, 1)

        if start_idx < 0:
            first_offset = int(center_padded_offsets[:, 0])
            left_padded_offsets = torch.from_numpy(  # TODO check logic is correct
                np.array([np.arange(first_offset - start_idx, first_offset, -1) % TICKS_PER_MEASURE])
            ).long().clone()

            left_padded_pitches = torch.from_numpy(
                np.array([self.start_pitch_symbol])
            ).long().clone().repeat(-start_idx, 1).transpose(0, 1)

            left_padded_durations = torch.from_numpy(
                np.array([self.start_duration_symbol])
            ).long().clone().repeat(-start_idx, 1).transpose(0, 1)

            # Not used --- added simply to keep the tensor size consistent
            left_padded_metadata = torch.from_numpy(
                np.array(self.METADATA_PADDING_SYMBOL)
            ).long().clone().repeat(-start_idx, self.METADATA_IDX_COUNT).transpose(0, 1)

            # TODO use chord for corresponding offset?
            left_padded_chord_pitches = torch.from_numpy(
                np.array([self.start_pitch_symbol])
            ).long().clone().repeat(-start_idx, self.chord_extension_count).transpose(0, 1)

            padded_offsets.append(left_padded_offsets)
            padded_pitches.append(left_padded_pitches)
            padded_durations.append(left_padded_durations)
            padded_metadata.append(left_padded_metadata)
            padded_chord_pitches.append(left_padded_chord_pitches)

        padded_offsets.append(center_padded_offsets)
        padded_pitches.append(center_padded_pitches)
        padded_durations.append(center_padded_durations)
        padded_metadata.append(center_padded_metadata)
        padded_chord_pitches.append(center_padded_chord_pitches)

        # Add right padding if necessary
        if end_idx > length:
            last_offset = int(center_padded_offsets[:, -1]) + int(center_padded_durations[:, -1])
            right_padded_offsets = torch.from_numpy(
                np.array([np.arange(last_offset, last_offset + end_idx - length, 1) % TICKS_PER_MEASURE])
            ).long().clone()

            right_padding_pitches = torch.from_numpy(
                np.array([self.end_pitch_symbol])
            ).long().clone().repeat(end_idx - length, 1).transpose(0, 1)

            right_padded_durations = torch.from_numpy(
                np.array([self.end_duration_symbol])
            ).long().clone().repeat(end_idx - length, 1).transpose(0, 1)

            # Not used --- added simply to keep the tensor size consistent
            right_padded_metadata = torch.from_numpy(
                np.array(self.METADATA_PADDING_SYMBOL)
            ).long().clone().repeat(end_idx - length, self.METADATA_IDX_COUNT).transpose(0, 1)

            # TODO use chord for corresponding offset?
            right_padded_chord_pitches = torch.from_numpy(
                np.array([self.end_pitch_symbol])
            ).long().clone().repeat(end_idx - length, self.chord_extension_count).transpose(0, 1)

            padded_offsets.append(right_padded_offsets)
            padded_pitches.append(right_padding_pitches)
            padded_durations.append(right_padded_durations)
            padded_metadata.append(right_padded_metadata)
            padded_chord_pitches.append(right_padded_chord_pitches)

        padded_offsets = torch.cat(padded_offsets, 1)
        padded_pitches = torch.cat(padded_pitches, 1)
        padded_durations = torch.cat(padded_durations, 1)
        padded_metadata = torch.cat(padded_metadata, 1)
        padded_chord_pitches = torch.cat(padded_chord_pitches, 1)

        padded_example = torch.cat([
            padded_offsets,
            padded_pitches,
            padded_durations,
            padded_metadata,
            padded_chord_pitches
        ], 0).transpose(0, 1).cuda()

        return padded_example[None, :, :]
