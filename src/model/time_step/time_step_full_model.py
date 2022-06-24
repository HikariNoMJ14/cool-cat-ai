import os
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from src.model import TimeStepChordModel
from src.generator import TimeStepFullGenerator
from src.utils import reverse_tensor

dir_path = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.join(dir_path, '..', '..', '..')


class TimeStepFullModel(TimeStepChordModel):
    # Input data semantics
    TENSOR_IDX_MAPPING = {
        'offset': 0,
        'improvised_pitch': 1,
        'improvised_attack': 2,
        'original_pitch': 3,
        'original_attack': 4,
        'metadata': 5,
        'chord_pitches_start': 6
    }

    FEATURES = {
        'past': [
            'offset',
            'improvised_pitch', 'improvised_attack',
            'original_pitch', 'original_attack'
        ],
        'present': [
            'offset',
            'original_pitch', 'original_attack',
            'metadata'
        ],
        'future': [
            'offset',
            'original_pitch', 'original_attack'
        ]
    }

    LABELS = [
        'improvised_pitch', 'improvised_attack'
    ]

    METRICS_LIST = [
        'pitch_loss', 'attack_loss',
        'pitch_top1', 'pitch_top3', 'pitch_top5',
        'attack_top1'
    ]

    PLOTTED_METRICS = ['loss', 'pitch_loss', 'attack_loss']

    GENERATOR_CLASS = TimeStepFullGenerator

    def __init__(self, dataset=None, logger=None, save_path=os.path.join(src_path, 'results'), **kwargs):
        super(TimeStepFullModel, self).__init__(dataset, logger, save_path, **kwargs)

        # Set model parameters
        self.start_attack_symbol = kwargs['start_attack_symbol']
        self.end_attack_symbol = kwargs['end_attack_symbol']

        self.attack_size = kwargs['attack_size']

        self.attack_loss_weight = kwargs['attack_loss_weight']

        self.attack_loss_function = nn.CrossEntropyLoss(
            weight=torch.Tensor([.1, .8, .05, .05])  # Focus on learning the occurrences of attack = 1
        )

        if kwargs['use_padding_idx'] and self.start_attack_symbol != self.end_attack_symbol:
            self.logger.warning('Start attack symbol and end attack symbol are different, '
                                'padding_idx will only act on start attack symbol')

        self.attack_encoder = nn.Sequential(
            nn.Embedding(
                self.attack_size,
                self.embedding_size,
                scale_grad_by_freq=True,
                padding_idx=self.start_attack_symbol if kwargs['use_padding_idx'] else None
            ),
            nn.Dropout(self.embedding_dropout_rate)
        )

        #  offset +
        #  improvised_pitch + improvised_attack +
        #  original_pitch + original_attack +
        #  chord_pitch * number_of_pitches
        past_lstm_input_size = self.embedding_size + \
                               self.embedding_size + self.embedding_size + \
                               self.embedding_size + self.embedding_size + \
                               self.embedding_size * self.chord_extension_count

        self.logger.debug(f'Model past LSTM input size: {past_lstm_input_size}')

        self.past_lstm = nn.LSTM(
            input_size=past_lstm_input_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            dropout=self.lstm_dropout_rate,
            batch_first=True
        )

        #  offset +
        #  original_pitch + original_attack +
        #  metadata +
        #  chord_pitch * number_of_pitches
        present_nn_input_size = self.embedding_size + \
                                self.embedding_size + self.embedding_size + \
                                self.embedding_size + \
                                self.embedding_size * self.chord_extension_count

        self.logger.debug(f'Model present LSTM input size: {present_nn_input_size}')

        self.present_nn = nn.Sequential(
            nn.Linear(present_nn_input_size, self.nn_hidden_size),
            nn.ReLU(),
            nn.Dropout(self.nn_dropout_rate),
            nn.Linear(self.nn_hidden_size, self.nn_output_size),
            nn.ReLU(),
            nn.Dropout(self.nn_dropout_rate)  # TODO check if performance degrades with many epochs
        )

        #  offset +
        #  original_pitch + original_attack +
        #  chord_pitch * number_of_pitches
        future_lstm_input_size = self.embedding_size + \
                                 self.embedding_size + self.embedding_size + \
                                 self.embedding_size * self.chord_extension_count

        self.logger.debug(f'Model future LSTM input size: {future_lstm_input_size}')

        self.future_lstm = nn.LSTM(
            input_size=future_lstm_input_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            dropout=self.lstm_dropout_rate,
            batch_first=True
        )

        merge_nn_input_size = self.lstm_hidden_size + self.nn_output_size + self.lstm_hidden_size
        merge_nn_output_size = self.embedding_size + self.embedding_size

        self.merge_nn = nn.Sequential(
            nn.Linear(merge_nn_input_size, self.nn_hidden_size),
            nn.ReLU(),
            nn.Dropout(self.nn_dropout_rate),
            nn.Linear(self.nn_hidden_size, merge_nn_output_size)
        )

        self.attack_decoder = nn.Linear(self.embedding_size, self.attack_size)

        # Tie attack encoder and attack weights
        self.attack_decoder.weight = self.attack_encoder[0].weight

    def prepare_past_lstm_input(self, past):
        # Extract features from past tensor
        past_offsets = self.extract_features(past, 'offset', 0)
        past_improvised_pitches = self.extract_features(past, 'improvised_pitch', 1)
        past_improvised_attacks = self.extract_features(past, 'improvised_attack', 2)
        past_original_pitches = self.extract_features(past, 'original_pitch', 3)
        past_original_attacks = self.extract_features(past, 'original_attack', 4)
        past_chord_pitches = self.extract_chords(past, (5, 12))

        # Encode past features
        past_offset_embedding = self.offset_encoder(past_offsets)
        past_improvised_pitch_embedding = self.pitch_encoder(past_improvised_pitches)
        past_original_pitch_embedding = self.pitch_encoder(past_original_pitches)
        past_chord_pitches_embedding = self.encode_chord_pitches(past_chord_pitches)
        past_improvised_attacks = self.attack_encoder(past_improvised_attacks)
        past_original_attacks = self.attack_encoder(past_original_attacks)

        return torch.cat([
            past_offset_embedding,
            past_improvised_pitch_embedding, past_improvised_attacks,
            past_original_pitch_embedding, past_original_attacks,
            past_chord_pitches_embedding
        ], 2)

    def prepare_present_nn_input(self, present):
        # Extract features from present tensor
        present_offsets = self.extract_features(present, 'offset', 0)
        present_original_pitches = self.extract_features(present, 'original_pitch', 1)
        present_original_attacks = self.extract_features(present, 'original_attack', 2)
        present_metadata = self.extract_features(present, 'metadata', 3)
        present_chord_pitches = self.extract_chords(present, (4, 11))

        # Encode present features
        present_offset_embedding = self.offset_encoder(present_offsets)
        present_original_pitch_embedding = self.pitch_encoder(present_original_pitches)
        present_original_attacks = self.attack_encoder(present_original_attacks)
        present_metadata_embedding = self.metadata_encoder(present_metadata)
        present_chord_pitches_embedding = self.encode_chord_pitches(present_chord_pitches)

        return torch.cat([
            present_offset_embedding,
            present_original_pitch_embedding, present_original_attacks,
            present_metadata_embedding,
            present_chord_pitches_embedding
        ], 2)

    def prepare_future_lstm_input(self, future):
        # Extract features from future tensor
        future_offsets = self.extract_features(future, 'offset', 0)
        future_original_pitches = self.extract_features(future, 'original_pitch', 1)
        future_original_attacks = self.extract_features(future, 'original_attack', 2)
        future_chord_pitches = self.extract_chords(future, (3, 10))

        # Encode future features
        future_offset_embedding = self.offset_encoder(future_offsets)
        future_original_pitch_embedding = self.pitch_encoder(future_original_pitches)
        future_original_attacks = self.attack_encoder(future_original_attacks)
        future_chord_pitches_embedding = self.encode_chord_pitches(future_chord_pitches)

        return torch.cat([
            future_offset_embedding,
            future_original_pitch_embedding, future_original_attacks,
            future_chord_pitches_embedding
        ], 2)

    def forward(self, features):
        past = features[0]
        present = features[1]
        future = features[2]

        self.cuda()

        # Past LSTM
        past_lstm_input = self.prepare_past_lstm_input(past)
        past_lstm_hidden = self.init_hidden(batch_size=past.size(0))
        past_lstm_output, past_lstm_hidden = self.past_lstm(past_lstm_input, past_lstm_hidden)
        past_lstm_output = past_lstm_output[:, -1, :]

        # Present NN
        present_nn_input = self.prepare_present_nn_input(present)
        present_nn_input = present_nn_input.view(present.size(0), -1)
        present_nn_output = self.present_nn(present_nn_input)

        # Future LSTM
        future_lstm_input = self.prepare_future_lstm_input(future)
        future_lstm_hidden = self.init_hidden(batch_size=future.size(0))
        future_lstm_output, future_lstm_hidden = self.future_lstm(future_lstm_input, future_lstm_hidden)
        future_lstm_output = future_lstm_output[:, -1, :]

        # Merge NN
        merge_nn_input = torch.cat([past_lstm_output, present_nn_output, future_lstm_output], 1)
        merge_nn_output = self.merge_nn(merge_nn_input)

        output_improvised_pitch = self.pitch_decoder(torch.sigmoid(merge_nn_output[:, :self.embedding_size]))
        output_improvised_attack = self.attack_decoder(torch.sigmoid(merge_nn_output[:, self.embedding_size:]))

        if self.normalize:
            output_improvised_pitch = F.normalize(output_improvised_pitch, p=2, dim=1)
            output_improvised_attack = F.normalize(output_improvised_attack, p=2, dim=1)

        return output_improvised_pitch, output_improvised_attack

    def prepare_examples(self, batch):
        batch_size, sequence_size, _ = batch.size()
        middle_tick = sequence_size // 2

        assert batch[:, middle_tick:middle_tick + 1, 1].eq(self.start_pitch_symbol).count_nonzero() == 0 and \
               batch[:, middle_tick:middle_tick + 1, 1].eq(self.end_pitch_symbol).count_nonzero() == 0
        assert batch[:, middle_tick:middle_tick + 1, 2].eq(self.start_attack_symbol).count_nonzero() == 0 and \
               batch[:, middle_tick:middle_tick + 1, 2].eq(self.end_attack_symbol).count_nonzero() == 0

        if self.start_pitch_symbol != self.end_pitch_symbol:
            assert batch[:, :middle_tick, 1].eq(self.end_pitch_symbol).count_nonzero() == 0
            assert batch[:, middle_tick + 1:, 1].eq(self.start_pitch_symbol).count_nonzero() == 0

        if self.start_attack_symbol != self.end_attack_symbol:
            assert batch[:, :middle_tick, 2].eq(self.end_attack_symbol).count_nonzero() == 0
            assert batch[:, middle_tick + 1:, 2].eq(self.start_attack_symbol).count_nonzero() == 0

        past_tensor_indices = list(range(0, 5)) + list(range(6, 13))
        past = batch[:, :middle_tick, past_tensor_indices]

        # Remove improvised pitch and attack from present tick
        present_tensor_indices = [0] + list(range(3, 13))
        present = batch[:, middle_tick:middle_tick + 1, present_tensor_indices]

        # Reverse sequence for future ticks
        reversed_tensor = reverse_tensor(
            batch[:, middle_tick + 1:, :], dim=1
        )
        # Remove improvised pitch and attack from future ticks
        # future_tensor_indices = [self.TENSOR_IDX_MAPPING[feature]
        #                          for feature in self.FEATURES['future']]
        # future_tensor_indices += self.chord_tensor_idx
        future_tensor_indices = [0] + [3, 4] + list(range(6, 13))
        future = self.mask_entry(
            reversed_tensor,
            future_tensor_indices,
            dim=2
        )
        future = batch[:, middle_tick:middle_tick + 1, future_tensor_indices]

        # Remove everything but improvised pitch and attack to get label
        label_tensor_indices = [1, 2]
        label = batch[:, middle_tick:middle_tick + 1:, label_tensor_indices]
        label = label.view(batch_size, -1)

        assert present[:, :, 1].eq(self.start_pitch_symbol).count_nonzero() == 0 and \
               present[:, :, 1].eq(self.end_pitch_symbol).count_nonzero() == 0
        assert label[:, 0].eq(self.start_pitch_symbol).count_nonzero() == 0 and \
               label[:, 0].eq(self.end_pitch_symbol).count_nonzero() == 0
        assert present[:, :, 2].eq(self.start_attack_symbol).count_nonzero() == 0 and \
               present[:, :, 2].eq(self.end_attack_symbol).count_nonzero() == 0
        assert label[:, 0].eq(self.start_attack_symbol).count_nonzero() == 0 and \
               label[:, 0].eq(self.end_attack_symbol).count_nonzero() == 0

        if self.start_pitch_symbol != self.end_pitch_symbol:
            assert past[:, :, 1].eq(self.end_pitch_symbol).count_nonzero() == 0
            assert future[:, :, 1].eq(self.start_pitch_symbol).count_nonzero() == 0
            assert past[:, :, 2].eq(self.end_attack_symbol).count_nonzero() == 0
            assert future[:, :, 2].eq(self.start_attack_symbol).count_nonzero() == 0

        return (past, present, future), label

    def create_padded_tensor(self, example, index):
        start_idx = index
        end_idx = index + self.sequence_size

        metadata_start = self.TENSOR_IDX_MAPPING['metadata']
        metadata_end = self.TENSOR_IDX_MAPPING['metadata'] + self.METADATA_IDX_COUNT

        chord_pitches_start = self.TENSOR_IDX_MAPPING['chord_pitches_start']
        chord_pitches_end = self.TENSOR_IDX_MAPPING['chord_pitches_start'] + self.chord_extension_count

        length = example.size(0)

        common_sliced_data = example[np.arange(start_idx, end_idx) % length]

        offsets = common_sliced_data[None, :, 0]
        original_pitches = common_sliced_data[None, :, 3]
        original_attacks = common_sliced_data[None, :, 4]

        metadata = common_sliced_data[:, metadata_start:metadata_end].transpose(0, 1)

        chord_pitches = torch.from_numpy(
            np.stack(
                common_sliced_data[:, chord_pitches_start:chord_pitches_end]
            )
        ).long().clone().transpose(0, 1)

        padded_improvised_pitches = []
        padded_improvised_attacks = []

        # Add left padding if necessary
        if start_idx < 0:
            left_improvised_pitches = torch.from_numpy(
                np.array([self.start_pitch_symbol])
            ).long().clone().repeat(-start_idx, 1).transpose(0, 1)

            left_improvised_attacks = torch.from_numpy(
                np.array([self.start_attack_symbol])
            ).long().clone().repeat(-start_idx, 1).transpose(0, 1)

            padded_improvised_pitches.append(left_improvised_pitches)
            padded_improvised_attacks.append(left_improvised_attacks)

        slice_start = start_idx if start_idx > 0 else 0
        slice_end = end_idx if end_idx < length else length

        sliced_data = example[slice_start:slice_end]

        center_improvised_pitches = sliced_data[None, :, 1]
        center_improvised_attacks = sliced_data[None, :, 2]

        padded_improvised_pitches.append(center_improvised_pitches)
        padded_improvised_attacks.append(center_improvised_attacks)

        # Add right padding if necessary
        if end_idx > length:
            right_improvised_pitches = torch.from_numpy(
                np.array([self.end_pitch_symbol])
            ).long().clone().repeat(end_idx - length, 1).transpose(0, 1)

            right_improvised_attacks = torch.from_numpy(
                np.array([self.end_attack_symbol])
            ).long().clone().repeat(end_idx - length, 1).transpose(0, 1)

            padded_improvised_pitches.append(right_improvised_pitches)
            padded_improvised_attacks.append(right_improvised_attacks)

        improvised_pitches = torch.cat(padded_improvised_pitches, 1)
        improvised_attacks = torch.cat(padded_improvised_attacks, 1)

        padded_example = torch.cat([
            offsets,
            improvised_pitches,
            improvised_attacks,
            original_pitches,
            original_attacks,
            metadata,
            chord_pitches
        ], 0).transpose(0, 1).cuda()

        return padded_example[None, :, :]
