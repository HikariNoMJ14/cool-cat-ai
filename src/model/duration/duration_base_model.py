import os
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from src.model.base import BaseModel
from src.generator import DurationBaseGenerator
from src.utils.constants import TICKS_PER_MEASURE

dir_path = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.join(dir_path, '..', '..', '..')


class DurationBaseModel(BaseModel):
    # Input data semantics
    TENSOR_IDX_MAPPING = {
        'original_flag': 0,
        'ticks': 1,
        'offset': 2,
        'pitch': 3,
        'duration': 4,
        'metadata': 5,
        'chord_pitches_start': 9
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

    GENERATOR_CLASS = DurationBaseGenerator

    def __init__(self, dataset=None, logger=None, save_path=os.path.join(src_path, 'results'), **kwargs):
        super(DurationBaseModel, self).__init__(dataset, logger, save_path, **kwargs)

        self.duration_to_ids = {}
        self.ids_to_durations = {}
        self.duration_size = 0

        # Set model parameters
        self.start_duration_symbol = kwargs['start_duration_symbol']
        self.end_duration_symbol = kwargs['end_duration_symbol']

        self.metadata_tensor_idx = list(range(
            self.TENSOR_IDX_MAPPING['metadata'],
            self.TENSOR_IDX_MAPPING['metadata'] + self.METADATA_IDX_COUNT
        ))

        self.setup_duration_mapping()

        self.duration_loss_weight = kwargs['duration_loss_weight']

        ignore_index = -100
        if self.start_pitch_symbol == self.end_pitch_symbol:
            ignore_index = self.start_pitch_symbol

        self.duration_loss_function = nn.CrossEntropyLoss(
            ignore_index=ignore_index
        )

        if kwargs['use_padding_idx'] and self.start_duration_symbol != self.end_duration_symbol:
            self.logger.warning('Start duration symbol and end duration symbol are different, '
                                'padding_idx will only act on start duration symbol')

        self.duration_encoder = nn.Sequential(
            nn.Embedding(
                self.duration_size,
                self.embedding_size,
                scale_grad_by_freq=True,
                padding_idx=self.start_duration_symbol if kwargs['use_padding_idx'] else None
            ),
            nn.Dropout(self.embedding_dropout_rate)
        )

        #  offset +
        #  pitch + duration
        past_lstm_input_size = self.embedding_size + self.embedding_size + self.embedding_size

        self.logger.debug(f'Model past LSTM input size: {past_lstm_input_size}')

        self.past_improvised_lstm = nn.LSTM(
            input_size=past_lstm_input_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            dropout=self.lstm_dropout_rate,
            batch_first=True
        )

        #  offset +
        #  metadata
        present_nn_input_size = self.embedding_size + \
                                self.embedding_size

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

    def setup_duration_mapping(self):
        all_durations = set({})

        for example in self.dataset.tensor_dataset:
            all_durations.update(set(example[:, self.TENSOR_IDX_MAPPING['duration']].tolist()))

        all_durations.add(self.start_duration_symbol)
        all_durations.add(self.end_duration_symbol)

        # Create dict to map duration values to ids
        self.duration_to_ids = dict((v, k) for k, v in enumerate(all_durations))

        # Create inverse dict to map ids to durations
        self.ids_to_durations = dict((k, v) for k, v in enumerate(all_durations))

        self.duration_size = len(self.duration_to_ids.keys())

    def convert_durations_to_ids(self, durations):
        return durations.cpu().apply_(lambda x: self.duration_to_ids[x]).cuda()

    def convert_ids_to_durations(self, ids):
        return ids.cpu().apply_(lambda x: self.ids_to_durations[x]).cuda()

    def prepare_past_lstm_input(self, past):
        # Extract features from past tensor
        past_offsets = self.extract_features(past, 'offset', 0)  # TODO fix
        past_pitches = self.extract_features(past, 'pitch', 1)
        past_durations = self.extract_features(past, 'duration', 2)
        past_durations = self.convert_durations_to_ids(past_durations)

        assert past_offsets.max() <= self.offset_size
        assert past_pitches.max() <= self.pitch_size
        assert past_durations.max() <= self.duration_size

        # Encode past offsets and pitches
        past_offset_embedding = self.offset_encoder(past_offsets)
        past_pitch_embedding = self.pitch_encoder(past_pitches)
        past_duration_embedding = self.duration_encoder(past_durations)

        return torch.cat([
            past_offset_embedding,
            past_pitch_embedding, past_duration_embedding
        ], 2)

    def prepare_present_nn_input(self, present):
        # Extract features from present tensor
        present_offsets = self.extract_features(present, 'offset', 0)  # TODO fix
        present_metadata = self.extract_features(present, 'metadata', 1)

        # Encode present offsets and pitches
        present_offset_embedding = self.offset_encoder(present_offsets)
        present_metadata_embedding = self.metadata_encoder(present_metadata)

        return torch.cat([
            present_offset_embedding,
            present_metadata_embedding
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
        output_improvised_duration = self.duration_decoder(torch.sigmoid(merge_nn_output[:, :self.embedding_size]))

        if self.normalize:
            output_improvised_pitch = F.normalize(output_improvised_pitch, p=2, dim=1)
            output_improvised_duration = F.normalize(output_improvised_duration, p=2, dim=1)

        return output_improvised_pitch, output_improvised_duration

    def prepare_examples(self, batch):
        improvised_batch = batch
        batch_size, sequence_size, _ = improvised_batch.size()
        middle_tick = sequence_size // 2

        assert improvised_batch[:, middle_tick:middle_tick + 1, 1].eq(self.start_pitch_symbol).count_nonzero() == 0
        assert improvised_batch[:, middle_tick:middle_tick + 1, 1].eq(self.end_pitch_symbol).count_nonzero() == 0

        if self.start_pitch_symbol != self.start_pitch_symbol:
            assert improvised_batch[:, :middle_tick, 1].eq(self.end_pitch_symbol).count_nonzero() == 0
            assert improvised_batch[:, middle_tick + 1:, 1].eq(self.start_pitch_symbol).count_nonzero() == 0

        past_improvised_tensor_indices = [self.TENSOR_IDX_MAPPING[feature]
                                          for feature in self.FEATURES['past_improvised']]
        past_improvised_tensor_indices = range(3)  # TODO fix!!!
        past_improvised = self.mask_entry(
            improvised_batch[:, :middle_tick, :],
            past_improvised_tensor_indices,
            dim=2
        )

        # Remove improvised pitch and duration from present tick
        present_tensor_indices = [self.TENSOR_IDX_MAPPING[feature]
                                  for feature in self.FEATURES['present']]
        present_tensor_indices = [0] + list(range(3, 10))  # TODO fix!!!
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
            improvised_batch[:, middle_tick:middle_tick + 1:, :],
            label_tensor_indices,
            dim=2
        )
        label = label.view(batch_size, -1)

        assert label[:, 0].eq(self.start_pitch_symbol).count_nonzero() == 0 and \
               label[:, 0].eq(self.end_pitch_symbol).count_nonzero() == 0

        if self.start_pitch_symbol != self.start_pitch_symbol:
            assert past_improvised[:, :, 1].eq(self.end_pitch_symbol).count_nonzero() == 0

        return (past_improvised, present), label

    def get_batch(self, dataset, batch_size):
        improvised_batch = []

        mid_point = self.sequence_size // 2

        for i in range(batch_size):
            # Choose random example from corpus
            random_example_idx = np.random.randint(0, len(dataset))
            chosen_example = dataset[random_example_idx]

            # Filter to only improvised notes
            improvised_mask = chosen_example[:, self.TENSOR_IDX_MAPPING['original_flag']] == 0
            improvised_chosen_example = chosen_example[improvised_mask]

            # Choose random note to predict
            improvised_random_idx = np.random.randint(-mid_point, len(improvised_chosen_example) - mid_point)

            # Pad improvised example
            improvised_padded_example = self.create_padded_tensor(improvised_chosen_example, improvised_random_idx)
            improvised_batch.append(improvised_padded_example)

        return torch.cat(improvised_batch, 0)

    def create_padded_tensor(self, example, index):
        start_idx = index
        end_idx = index + self.sequence_size

        metadata_start = self.TENSOR_IDX_MAPPING['metadata']
        metadata_end = self.TENSOR_IDX_MAPPING['metadata'] + self.METADATA_IDX_COUNT

        length = example.size(0)

        padded_offsets = []
        padded_pitches = []
        padded_durations = []
        padded_metadata = []

        slice_start = start_idx if start_idx > 0 else 0
        slice_end = end_idx if end_idx < length else length

        sliced_data = example[slice_start:slice_end]

        center_padded_offsets = sliced_data[None, :, self.TENSOR_IDX_MAPPING['offset']]
        center_padded_pitches = sliced_data[None, :, self.TENSOR_IDX_MAPPING['pitch']]
        center_padded_durations = sliced_data[None, :, self.TENSOR_IDX_MAPPING['duration']]
        center_padded_metadata = sliced_data[:, metadata_start:metadata_end].transpose(0, 1)

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

            left_padded_metadata = torch.from_numpy(
                np.array(self.METADATA_SYMBOL)
            ).long().clone().repeat(-start_idx, self.METADATA_IDX_COUNT).transpose(0, 1)

            padded_offsets.append(left_padded_offsets)
            padded_pitches.append(left_padded_pitches)
            padded_durations.append(left_padded_durations)
            padded_metadata.append(left_padded_metadata)

        padded_offsets.append(center_padded_offsets)
        padded_pitches.append(center_padded_pitches)
        padded_durations.append(center_padded_durations)
        padded_metadata.append(center_padded_metadata)

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

            right_padded_metadata = torch.from_numpy(
                np.array(self.METADATA_SYMBOL)
            ).long().clone().repeat(end_idx - length, self.METADATA_IDX_COUNT).transpose(0, 1)

            padded_offsets.append(right_padded_offsets)
            padded_pitches.append(right_padding_pitches)
            padded_durations.append(right_padded_durations)
            padded_metadata.append(right_padded_metadata)

        padded_offsets = torch.cat(padded_offsets, 1)
        padded_pitches = torch.cat(padded_pitches, 1)
        padded_durations = torch.cat(padded_durations, 1)
        padded_metadata = torch.cat(padded_metadata, 1)

        padded_example = torch.cat([
            padded_offsets,
            padded_pitches,
            padded_durations,
            padded_metadata
        ], 0).transpose(0, 1).cuda()

        return padded_example[None, :, :]

    def normalize_embeddings(self):
        super().normalize_embeddings()

        self.duration_decoder.weight.data = F.normalize(self.duration_encoder.weight, p=2, dim=1)

    def loss_function(self, prediction, label):
        output_pitch = prediction[0]
        output_duration = prediction[1]
        pitch_loss = self.pitch_loss_function(output_pitch, label[:, 0])
        duration_loss = self.duration_loss_function(output_duration, self.convert_durations_to_ids(label[:, 1]))

        pitch_top1, \
        pitch_top3, \
        pitch_top5 = self.accuracy(
            output_pitch,
            label[:, 0].contiguous(),
            topk=(1, 3, 5)
        )

        duration_top1, \
        duration_top3, \
        duration_top5 = self.accuracy(
            output_duration,
            label[:, 1].contiguous(),
            topk=(1, 3, 5)
        )

        metrics = {
            'pitch_loss': pitch_loss, 'duration_loss': duration_loss,
            'pitch_top1': pitch_top1, 'pitch_top3': pitch_top3, 'pitch_top5': pitch_top5,
            'duration_top1': duration_top1, 'duration_top3': duration_top3, 'duration_top5': duration_top5
        }

        total_loss = self.pitch_loss_weight * pitch_loss + \
                     self.duration_loss_weight * duration_loss

        return total_loss, metrics
