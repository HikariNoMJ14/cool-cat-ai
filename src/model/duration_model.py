import os
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from src.model import Model

dir_path = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.join(dir_path, '..', '..')


class DurationModel(Model):
    # Input data semantics
    TENSOR_IDX_MAPPING = {
        'offset': 0,
        'pitch': 1,
        'duration': 2,
        'chord_pitches_start': 3
    }
    
    FEATURES = {
        'past_original': [
            'offset',
            'original_pitch', 'original_duration'
        ],
        'past_improvised': [
            'offset',
            'improvised_pitch', 'improvised_duration'
        ],
        # TODO Add metadata
        'present': [
            'offset',
            'original_pitch', 'original_duration'
        ],
        'future': [
            'offset',
            'original_pitch', 'original_duration'
        ]
    }

    LABELS = [
        'improvised_pitch', 'improvised_duration'
    ]

    METRICS_LIST = [
        'pitch_loss', 'duration_loss',
        'pitch_top1', 'pitch_top3', 'pitch_top5',
        'duration_top1', 'duration_top3', 'duration_top5'
    ]

    PLOTTED_METRICS = ['loss', 'ppl', 'pitch_loss', 'duration_loss']

    def __init__(self, dataset=None, logger=None, save_path=os.path.join(src_path, 'results'), **kwargs):
        super(DurationModel, self).__init__(dataset, logger, save_path, **kwargs)

        # Set model parameters
        self.duration_size = kwargs['duration_size']

        self.duration_loss_weight = kwargs['duration_loss_weight']

        self.duration_loss_function = nn.CrossEntropyLoss()

        self.duration_encoder = nn.Sequential(
            nn.Embedding(self.duration_size, self.embedding_size, scale_grad_by_freq=True),
            nn.Dropout(self.embedding_dropout_rate)
        )
        # TODO add padding_idx for 128 (rest)? what about START_SYMBOL and END_SYMBOL?

        #  offset +
        #  pitch + duration +
        #  chord_pitch * number_of_pitches
        past_lstm_input_size = self.embedding_size + \
                               self.embedding_size + self.duration_size + \
                               self.embedding_size * self.chord_extension_count \
            if self.chord_encoding_type != 'compressed' \
            else 12

        self.logger.debug(f'Model past LSTM input size: {past_lstm_input_size}')

        self.past_improvised_lstm = nn.LSTM(
            input_size=past_lstm_input_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            dropout=self.lstm_dropout_rate,
            batch_first=True
        )

        self.past_original_lstm = nn.LSTM(
            input_size=past_lstm_input_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            dropout=self.lstm_dropout_rate,
            batch_first=True
        )

        #  offset +
        #  original_pitch + original_duration +
        #  metadata +
        #  chord_pitch * number_of_pitches
        present_nn_input_size = self.embedding_size + \
                                self.embedding_size + self.duration_size + \
                                self.metadata_size + \
                                self.embedding_size * self.chord_extension_count \
            if self.chord_encoding_type != 'compressed' \
            else 12

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
        #  original_pitch + original_duration +
        #  chord_pitch * number_of_pitches
        future_lstm_input_size = self.embedding_size + \
                                 self.embedding_size + self.duration_size + \
                                 self.embedding_size * self.chord_extension_count \
            if self.chord_encoding_type != 'compressed' \
            else 12

        self.logger.debug(f'Model future LSTM input size: {future_lstm_input_size}')

        self.future_lstm = nn.LSTM(
            input_size=future_lstm_input_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            dropout=self.lstm_dropout_rate,
            batch_first=True
        )

        merge_nn_input_size = self.lstm_hidden_size + self.nn_output_size + self.lstm_hidden_size
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
        past_offsets = self.extract_features(past, 'offset', 0)
        past_pitches = self.extract_features(past, 'pitch', 1)
        past_durations = self.extract_features(past, 'duration', 2)
        past_chord_pitches = self.extract_chords(past, (5, 12))

        # Encode past offsets and pitches
        past_offset_embedding = self.offset_encoder(past_offsets)
        past_pitch_embedding = self.pitch_encoder(past_pitches)
        past_chord_pitches_embedding = self.encode_chord_pitches(past_chord_pitches)
        past_durations = past_durations[:, :, None]

        return torch.cat([
            past_offset_embedding,
            past_pitch_embedding, past_durations,
            past_chord_pitches_embedding
        ], 2)

    def prepare_present_nn_input(self, present):
        # Extract features from present tensor
        present_offsets = self.extract_features(present, 'offset', 0)
        present_original_pitches = self.extract_features(present, 'original_pitch', 1)
        present_original_durations = self.extract_features(present, 'original_duration', 2)
        present_chord_pitches = self.extract_chords(present, (3, 10))

        # Encode present offsets and pitches
        present_offset_embedding = self.offset_encoder(present_offsets)
        present_original_pitch_embedding = self.pitch_encoder(present_original_pitches)
        present_chord_pitches_embedding = self.encode_chord_pitches(present_chord_pitches)
        present_original_durations = present_original_durations[:, :, None]

        return torch.cat([
            present_offset_embedding,
            present_original_pitch_embedding, present_original_durations,
            present_chord_pitches_embedding
        ], 2)

    def prepare_future_lstm_input(self, future):
        # Extract features from future tensor
        future_offsets = self.extract_features(future, 'offset', 0)
        future_original_pitches = self.extract_features(future, 'original_pitch', 1)
        future_original_durations = self.extract_features(future, 'original_duration', 2)
        future_chord_pitches = self.extract_chords(future, (3, 10))

        # Encode future offsets and pitches
        future_offset_embedding = self.offset_encoder(future_offsets)
        future_original_pitch_embedding = self.pitch_encoder(future_original_pitches)
        future_chord_pitches_embedding = self.encode_chord_pitches(future_chord_pitches)
        future_original_durations = future_original_durations[:, :, None]

        return torch.cat([
            future_offset_embedding,
            future_original_pitch_embedding, future_original_durations,
            future_chord_pitches_embedding
        ], 2)

    def forward(self, features):
        past_original = features[0]
        past_improvised = features[1]
        present = features[2]
        future = features[3]

        self.cuda()

        # Past Original LSTM
        past_original_lstm_input = self.prepare_past_lstm_input(past_original)
        past_original_lstm_hidden = self.init_hidden(batch_size=past_original.size(0))
        past_original_lstm_output, \
        past_original_lstm_hidden = self.past_original_lstm(
            past_original_lstm_input,
            past_original_lstm_hidden
        )
        past_original_lstm_output = past_original_lstm_output[:, -1, :]

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

        # Future LSTM
        future_lstm_input = self.prepare_future_lstm_input(future)
        future_lstm_hidden = self.init_hidden(batch_size=future.size(0))
        future_lstm_output, future_lstm_hidden = self.future_lstm(future_lstm_input, future_lstm_hidden)
        future_lstm_output = future_lstm_output[:, -1, :]

        # Merge NN
        merge_nn_input = torch.cat([
            past_original_lstm_output,
            past_improvised_lstm_output,
            present_nn_output,
            future_lstm_output
        ], 1)
        merge_nn_output = self.merge_nn(merge_nn_input)

        output_improvised_pitch = self.pitch_decoder(torch.sigmoid(merge_nn_output[:, :self.embedding_size]))
        output_improvised_duration = self.duration_decoder(torch.sigmoid(merge_nn_output[:, :self.embedding_size]))

        if self.normalize:
            output_improvised_pitch = F.normalize(output_improvised_pitch, p=2, dim=1)
            output_improvised_duration = F.normalize(output_improvised_duration, p=2, dim=1)

        return output_improvised_pitch, output_improvised_duration

    def prepare_examples(self, batch):
        batch_size, sequence_size, _ = batch.size()
        middle_tick = sequence_size // 2

        assert batch[:, middle_tick:middle_tick + 1, 1].eq(self.start_symbol).count_nonzero() == 0
        assert batch[:, middle_tick:middle_tick + 1, 1].eq(self.end_symbol).count_nonzero() == 0
        assert batch[:, middle_tick, 1].eq(self.end_symbol).count_nonzero() == 0
        assert batch[:, middle_tick + 1:, 1].eq(self.start_symbol).count_nonzero() == 0

        past_original_tensor_indices = [self.TENSOR_IDX_MAPPING[feature]
                               for feature in self.FEATURES['past_original']]
        past_original_tensor_indices += self.chord_tensor_idx
        past_original = self.mask_entry(
            batch[:, :middle_tick, :],
            past_original_tensor_indices,
            dim=2
        )
        
        past_improvised_tensor_indices = [self.TENSOR_IDX_MAPPING[feature]
                               for feature in self.FEATURES['past_improvised']]
        past_improvised_tensor_indices += self.chord_tensor_idx
        past_improvised = self.mask_entry(
            batch[:, :middle_tick, :],
            past_improvised_tensor_indices,
            dim=2
        )

        # Remove improvised pitch and duration from present tick
        present_tensor_indices = [self.TENSOR_IDX_MAPPING[feature]
                                  for feature in self.FEATURES['present']]
        present_tensor_indices += self.chord_tensor_idx
        present = self.mask_entry(
            batch[:, middle_tick:middle_tick + 1, :],
            present_tensor_indices,
            dim=2
        )

        # Reverse sequence for future ticks
        reversed_tensor = self.reverse_tensor(
            batch[:, middle_tick + 1:, :], dim=1
        )
        # Remove improvised pitch and duration from future ticks
        future_tensor_indices = [self.TENSOR_IDX_MAPPING[feature]
                                 for feature in self.FEATURES['future']]
        future_tensor_indices += self.chord_tensor_idx
        future = self.mask_entry(
            reversed_tensor,
            future_tensor_indices,
            dim=2
        )

        # Remove everything but improvised pitch and duration to get label
        label_tensor_indices = [self.TENSOR_IDX_MAPPING[feature]
                                for feature in self.LABELS]
        label = self.mask_entry(
            batch[:, middle_tick:middle_tick + 1:, :],
            label_tensor_indices,
            dim=2
        )
        label = label.view(batch_size, -1)

        assert past_original.eq(self.end_symbol).count_nonzero() == 0
        assert past_improvised.eq(self.end_symbol).count_nonzero() == 0
        assert present.eq(self.start_symbol).count_nonzero() == 0 and present.eq(self.end_symbol).count_nonzero() == 0
        assert future.eq(self.start_symbol).count_nonzero() == 0
        assert label.eq(self.start_symbol).count_nonzero() == 0 and label.eq(self.end_symbol).count_nonzero() == 0

        return (past_original, past_improvised, present, future), label

    def create_padded_tensor(self, example, index):
        start_idx = index
        end_idx = index + self.sequence_size

        length = example.size(0)

        common_sliced_data = example[np.arange(start_idx, end_idx) % length]

        offsets = common_sliced_data[None, :, 0]
        original_pitches = common_sliced_data[None, :, 1]
        original_durations = common_sliced_data[None, :, 2]

        chord_pitches = torch.from_numpy(
            np.stack(
                common_sliced_data[:, 5:12]
            )
        ).long().clone().transpose(0, 1)

        padded_improvised_pitches = []
        padded_improvised_durations = []

        # Add left padding if necessary
        if start_idx < 0:
            left_improvised_pitches = torch.from_numpy(
                np.array([self.start_symbol])
            ).long().clone().repeat(-start_idx, 1).transpose(0, 1)

            left_improvised_durations = torch.from_numpy(
                np.array([0])
            ).long().clone().repeat(-start_idx, 1).transpose(0, 1)

            padded_improvised_pitches.append(left_improvised_pitches)
            padded_improvised_durations.append(left_improvised_durations)

        slice_start = start_idx if start_idx > 0 else 0
        slice_end = end_idx if end_idx < length else length

        sliced_data = example[slice_start:slice_end]

        center_improvised_pitches = sliced_data[None, :, 1]
        center_improvised_durations = sliced_data[None, :, 2]

        padded_improvised_pitches.append(center_improvised_pitches)
        padded_improvised_durations.append(center_improvised_durations)

        # Add right padding if necessary
        if end_idx > length:
            right_improvised_pitches = torch.from_numpy(
                np.array([self.end_symbol])
            ).long().clone().repeat(end_idx - length, 1).transpose(0, 1)

            right_improvised_durations = torch.from_numpy(
                np.array([0])
            ).long().clone().repeat(end_idx - length, 1).transpose(0, 1)

            padded_improvised_pitches.append(right_improvised_pitches)
            padded_improvised_durations.append(right_improvised_durations)

        improvised_pitches = torch.cat(padded_improvised_pitches, 1)
        improvised_durations = torch.cat(padded_improvised_durations, 1)

        padded_example = torch.cat([
            offsets,
            improvised_pitches,
            improvised_durations,
            original_pitches,
            original_durations,
            chord_pitches
        ], 0).transpose(0, 1).cuda()

        return padded_example[None, :, :]

    def normalize_embeddings(self):
        super().normalize_embeddings()

        self.duration_decoder.weight.data = F.normalize(self.duration_encoder.weight, p=2, dim=1)

    def loss_function(self, prediction, label):
        output_pitch = prediction[0]
        output_duration = prediction[1]
        pitch_loss = self.pitch_loss_function(output_pitch, label[:, 0])
        duration_loss = self.duration_loss_function(output_duration, label[:, 1].float())

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
