import os
import logging
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from src.dataset import Dataset
from src.utils import Metric

dir_path = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.join(dir_path, '..', '..')


class MonoTimeStepModel(nn.Module):
    # Input data semantics
    TENSOR_IDX_MAPPING = {
        'offset': 0,
        'improvised_pitch': 1,
        'improvised_attack': 2,
        'original_pitch': 3,
        'original_attack': 4,
        'chord_pitches_start': 5
    }

    FEATURES = {
        'past': [
            'offset',
            'improvised_pitch', 'improvised_attack',
            'original_pitch', 'original_attack'
        ],
        # TODO Add metadata
        'present': [
            'offset',
            'original_pitch', 'original_attack'
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
    VOLATILE = False
    LOG_INTERVAL = 500

    def __init__(self, dataset: Dataset, logger, save_path=os.path.join(src_path, 'results'), **kwargs):
        super(MonoTimeStepModel, self).__init__()

        self.name = ''
        self.save_dir = os.path.join(src_path, 'results')

        self.dataset = dataset
        self.logger = logger
        self.save_path = save_path

        self.logger.info('--- Init Model ---')

        # Set model parameters
        self.offset_size = kwargs['offset_size']
        self.pitch_size = kwargs['pitch_size']
        self.attack_size = kwargs['attack_size']
        self.metadata_size = kwargs['metadata_size']
        self.embedding_size = kwargs['embedding_size']
        self.embedding_dropout_rate = kwargs['embedding_dropout_rate']
        self.lstm_num_layers = kwargs['lstm_num_layers']
        self.lstm_hidden_size = kwargs['lstm_hidden_size']
        self.lstm_dropout_rate = kwargs['lstm_dropout_rate']
        self.nn_hidden_size = kwargs['nn_hidden_size']
        self.nn_output_size = kwargs['nn_output_size']
        self.nn_dropout_rate = kwargs['nn_dropout_rate']
        self.normalize = kwargs['normalize']
        self.gradient_clipping = kwargs['gradient_clipping']
        self.pitch_loss_weight = kwargs['pitch_loss_weight']
        self.attack_loss_weight = kwargs['attack_loss_weight']

        self.sequence_size = dataset.sequence_size
        self.chord_extension_count = dataset.chord_extension_count
        self.chord_encoding_type = dataset.chord_encoding_type

        self.chord_tensor_idx = list(range(
            self.TENSOR_IDX_MAPPING['chord_pitches_start'],
            self.TENSOR_IDX_MAPPING['chord_pitches_start'] + self.chord_extension_count,
        ))

        self.pitch_loss_function = nn.CrossEntropyLoss(
            # ignore_index=129 # TODO ignore padding value? what about 130?
        )
        self.attack_loss_function = nn.BCEWithLogitsLoss()

        # TODO change to locked_drop?
        self.offset_encoder = nn.Sequential(
            nn.Embedding(self.offset_size, self.embedding_size),
            nn.Dropout(self.embedding_dropout_rate)
        )
        self.pitch_encoder = nn.Sequential(
            nn.Embedding(self.pitch_size, self.embedding_size, scale_grad_by_freq=True),
            nn.Dropout(self.embedding_dropout_rate)
        )
        # TODO add padding_idx for 128 (rest)? what about START_SYMBOL and END_SYMBOL?

        #  offset +
        #  improvised_pitch + improvised_attack +
        #  original_pitch + original_attack +
        #  chord_pitch * number_of_pitches
        past_lstm_input_size = self.embedding_size + \
                               self.embedding_size + self.attack_size + \
                               self.embedding_size + self.attack_size + \
                               self.embedding_size * self.chord_extension_count \
            if self.chord_encoding_type != 'compressed' \
            else 12

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
                                self.embedding_size + self.attack_size + \
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
        #  original_pitch + original_attack +
        #  chord_pitch * number_of_pitches
        future_lstm_input_size = self.embedding_size + \
                                 self.embedding_size + self.attack_size + \
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
        merge_nn_output_size = self.embedding_size + self.attack_size

        self.merge_nn = nn.Sequential(
            nn.Linear(merge_nn_input_size, self.nn_hidden_size),
            nn.ReLU(),
            nn.Dropout(self.nn_dropout_rate),
            nn.Linear(self.nn_hidden_size, merge_nn_output_size)
        )

        self.pitch_decoder = nn.Sequential(
            nn.Linear(self.embedding_size, self.pitch_size),
            nn.Dropout(self.embedding_dropout_rate)
        )

        # Tie pitch encoder and decoder weights
        self.pitch_decoder[0].weight = self.pitch_encoder[0].weight

    # From DeepBach
    def init_hidden(self, batch_size):
        hidden = (
            torch.randn(self.lstm_num_layers, batch_size, self.lstm_hidden_size).cuda(),
            torch.randn(self.lstm_num_layers, batch_size, self.lstm_hidden_size).cuda()
        )
        return hidden

    def encode_chord_pitches(self, chord_pitches):
        chord_pitches_flat = chord_pitches.view(-1)
        chord_pitches_embedding = self.pitch_encoder(chord_pitches_flat) \
            .view(chord_pitches.size(0), chord_pitches.size(1), -1)

        return chord_pitches_embedding

    # TODO refactor - move impro_pitch and attack to end of tensor -> remove idx param
    def extract_features(self, tensor, feature_name, idx):
        feature_tensor = tensor[:, :, idx]

        return torch.squeeze(feature_tensor).contiguous().view(tensor.size(0), -1)

    # TODO refactor - move impro_pitch and attack to end of tensor -> remove idx param
    def extract_chords(self, tensor, idx):
        chord_tensor = tensor[:, :, idx[0]:idx[1] + 1]

        return torch.squeeze(chord_tensor).contiguous().view(tensor.size(0), tensor.size(1), -1)

    def prepare_past_lstm_input(self, past):
        # Extract features from past tensor
        past_offsets = self.extract_features(past, 'offset', 0)
        past_improvised_pitches = self.extract_features(past, 'improvised_pitch', 1)
        past_improvised_attacks = self.extract_features(past, 'improvised_attack', 2)
        past_original_pitches = self.extract_features(past, 'original_pitch', 3)
        past_original_attacks = self.extract_features(past, 'original_attack', 4)
        past_chord_pitches = self.extract_chords(past, (5, 12))

        # Encode past offsets and pitches
        past_offset_embedding = self.offset_encoder(past_offsets)
        past_improvised_pitch_embedding = self.pitch_encoder(past_improvised_pitches)
        past_original_pitch_embedding = self.pitch_encoder(past_original_pitches)
        past_chord_pitches_embedding = self.encode_chord_pitches(past_chord_pitches)
        past_improvised_attacks = past_improvised_attacks[:, :, None]
        past_original_attacks = past_original_attacks[:, :, None]

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
        present_chord_pitches = self.extract_chords(present, (3, 10))

        # Encode present offsets and pitches
        present_offset_embedding = self.offset_encoder(present_offsets)
        present_original_pitch_embedding = self.pitch_encoder(present_original_pitches)
        present_chord_pitches_embedding = self.encode_chord_pitches(present_chord_pitches)
        present_original_attacks = present_original_attacks[:, :, None]

        return torch.cat([
            present_offset_embedding,
            present_original_pitch_embedding, present_original_attacks,
            present_chord_pitches_embedding
        ], 2)

    def prepare_future_lstm_input(self, future):
        # Extract features from future tensor
        future_offsets = self.extract_features(future, 'offset', 0)
        future_original_pitches = self.extract_features(future, 'original_pitch', 1)
        future_original_attacks = self.extract_features(future, 'original_attack', 2)
        future_chord_pitches = self.extract_chords(future, (3, 10))

        # Encode future offsets and pitches
        future_offset_embedding = self.offset_encoder(future_offsets)
        future_original_pitch_embedding = self.pitch_encoder(future_original_pitches)
        future_chord_pitches_embedding = self.encode_chord_pitches(future_chord_pitches)
        future_original_attacks = future_original_attacks[:, :, None]

        return torch.cat([
            future_offset_embedding,
            future_original_pitch_embedding, future_original_attacks,
            future_chord_pitches_embedding
        ], 2)

    def forward(self, past, present, future):
        self.cuda()

        # TODO add dropout(s) ?

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
        output_improvised_attack = merge_nn_output[:, -self.attack_size:].view(-1)

        if self.normalize:
            output_improvised_pitch = F.normalize(output_improvised_pitch, p=2, dim=1)
            output_improvised_attack = F.normalize(output_improvised_attack, p=2, dim=1)

        return output_improvised_pitch, output_improvised_attack

    def normalize_embeddings(self):
        self.encode_pitch.weight.data = F.normalize(self.encode_pitch.weight, p=2, dim=1)
        self.encode_duration.weight.data = F.normalize(self.encode_duration.weight, p=2, dim=1)

    def train_and_eval(self, batch_size, num_epochs, optimizer, scheduler, seed, callback):
        self.name = f'{self.dataset.name}_batchsize_{batch_size}_seed_{seed}'

        checkpoint_path = os.path.join(self.save_path, self.name + '.pt')
        log_path = os.path.join(self.save_path, 'log.log')
        results_path = os.path.join(self.save_path, 'results.csv')

        fileHandler = logging.FileHandler(log_path)
        fileHandler.setFormatter(self.logger.handlers[0].formatter)
        self.logger.addHandler(fileHandler)

        self.logger.info(f'--- Training ---')

        num_params = 0
        for p in self.parameters():
            num_params += p.numel()

        self.logger.info(f'Num Params: {num_params}')
        self.logger.info(f'Batch Size: {batch_size}')
        self.logger.info(f'Sequence Size: {self.sequence_size}')
        self.logger.info(f'Seed: {seed}')

        self.logger.info(f"Model checkpoint path to {checkpoint_path}")

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        total_start_time = time.time()
        best_valid_accuracy = None
        training_results = []

        try:
            for epoch in range(1, num_epochs + 1):
                epoch_start_time = time.time()

                train_dataloader, \
                val_dataloader, \
                test_dataloader = self.dataset.data_loaders(
                    batch_size=batch_size,
                    split=(.85, .15, 0),
                    seed=seed
                )

                if epoch == 1:
                    self.logger.info(f'Number of training examples: {len(train_dataloader)}')
                    self.logger.info(f'Number of training examples: {len(val_dataloader)}')

                self.logger.info(f'--- Epoch {epoch} ---')

                train_loss, train_metrics = self.loss_and_accuracy(
                    dataloader=train_dataloader,
                    optimizer=optimizer,
                    phase='train',
                    batch_size=batch_size
                )

                valid_loss, valid_metrics = self.loss_and_accuracy(
                    dataloader=val_dataloader,
                    optimizer=optimizer,
                    phase='test',
                    batch_size=batch_size
                )

                training_str = ' | '.join([f'train_{k}: {v:5.2f}' for k, v in train_metrics.items()])
                validation_str = ' | '.join([f'valid_{k}: {v:5.2f}' for k, v in valid_metrics.items()])
                self.logger.info(
                    f'End of epoch {epoch:3d} '
                    f'Time: {(time.time() - epoch_start_time):5.2f}s '
                )
                self.logger.info(
                    f'Train loss {train_loss:5.2f} '
                    f'Train ppl {np.exp(train_loss):8.2f} '
                    f'| {training_str}'
                )
                self.logger.info(
                    f'Valid loss {valid_loss:5.2f} '
                    f'Valid ppl {np.exp(valid_loss):8.2f} '
                    f'| {validation_str}'
                )

                epoch_results = dict({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_ppl': np.exp(train_loss),
                    'valid_loss': valid_loss,
                    'valid_ppl': np.exp(valid_loss)
                })
                for name, value in train_metrics.items():
                    epoch_results['train_' + name] = value
                for name, value in valid_metrics.items():
                    epoch_results['valid_' + name] = value
                training_results.append(epoch_results)

                # Save the model if the validation loss is the best we've seen so far.
                average_valid_accuracy = valid_metrics['pitch_top1']
                if not best_valid_accuracy or average_valid_accuracy < best_valid_accuracy:
                    with open(checkpoint_path.replace('.pt', '_best_val.pt'), 'wb') as f:
                        torch.save(self, f)
                        best_valid_accuracy = average_valid_accuracy

                if epoch % 50 == 0:
                    with open(checkpoint_path.replace('.pt', f'_e{epoch}.pt'), 'wb') as f:
                        torch.save(self, f)
                        best_valid_accuracy = average_valid_accuracy

                scheduler.step()

            self.logger.info('--- End of Training ---')
            self.logger.info(f'Time: {(time.time() - total_start_time):5.2f}s')

            training_results = pd.DataFrame.from_dict(training_results)
            training_results.to_csv(results_path)

            callback('FINISHED', training_results)

            # Plot training results
            for metric in ['loss', 'ppl', 'pitch_loss', 'attack_loss']:
                fig, ax = plt.subplots(nrows=1, ncols=1)
                for phase in ['train', 'valid']:
                    ax.plot(training_results[f'{phase}_{metric}'], label=f'{phase}_{metric.replace("_", " ")}')
                fig.legend(loc="upper right")
                fig.savefig(os.path.join(self.save_path, f'{metric}.png'))
                plt.close(fig)

            with open(checkpoint_path, 'wb') as f:
                torch.save(self, f)

        except KeyboardInterrupt:
            self.logger.info('--- Training stopped ---')
            self.logger.info(f'Time: {(time.time() - total_start_time):5.2f}s')

            checkpoint_path = os.path.join(self.save_path, self.name + '_early_stop.pt')

            training_results = pd.DataFrame.from_dict(training_results)
            training_results.to_csv(results_path)

            callback('KILLED', training_results)

            if len(training_results) > 0:
                # Plot training results
                for metric in ['loss', 'ppl', 'pitch_loss', 'attack_loss']:
                    fig, ax = plt.subplots(nrows=1, ncols=1)
                    for phase in ['train', 'valid']:
                        ax.plot(training_results[f'{phase}_{metric}'], label=f'{phase}_{metric.replace("_", " ")}')
                    fig.legend(loc="upper right")
                    fig.savefig(os.path.join(self.save_path, f'{metric}.png'))
                    plt.close(fig)

                with open(checkpoint_path, 'wb') as f:
                    torch.save(self, f)

    def loss_and_accuracy(self, dataloader, phase, optimizer, batch_size):
        if phase == 'train':
            self.train()
        elif phase == 'eval' or phase == 'test':
            self.eval()

        loss = Metric()
        metrics = {k: Metric() for k in self.METRICS_LIST}
        logging_loss = Metric()
        logging_metrics = {k: Metric() for k in self.METRICS_LIST}
        start_time = time.time()

        num_batches = len(self.dataset.tensor_dataset) // batch_size

        for i, batch in enumerate(dataloader):
            batch = Variable(batch[0], volatile=self.VOLATILE).long().cuda()

            past, present, future, label = self.prepare_examples(batch)
            output_pitch, output_attack = self(past, present, future)
            current_loss, current_metrics = self.loss_function(output_pitch, output_attack, label)

            if phase == 'train':
                optimizer.zero_grad()
                current_loss.backward()

                grad_norm = nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clipping)
                optimizer.step()

                if self.normalize:
                    self.normalize_embeddings()

            loss.update(float(current_loss), batch_size)
            logging_loss.update(float(current_loss), batch_size)
            for name in metrics.keys():
                metrics[name].update(float(current_metrics[name]))
                logging_metrics[name].update(float(current_metrics[name]), self.sequence_size)

            if phase == 'train':
                if i % self.LOG_INTERVAL == 0 and i > 0:
                    cur_loss = logging_loss.avg
                    elapsed = time.time() - start_time
                    metric_str = ' | '.join([f'{k}: {v.avg:5.2f}' for k, v in logging_metrics.items()])

                    self.logger.info(
                        f'| {int(100 * i / num_batches):3d}% '
                        # f'| ms/batch {(elapsed * 1000 / self.LOG_INTERVAL):7.2f} '
                        f'| loss {cur_loss:5.2f} '
                        f'| ppl {np.exp(cur_loss):6.2f} '
                        f'| grad_norm {grad_norm:5.2f} '
                        f'| {metric_str}'
                    )

                    logging_loss.reset()
                    for name in metrics.keys():
                        logging_metrics[name].reset()
                    start_time = time.time()

        avg_loss = loss.avg
        avg_metrics = {}
        for name in metrics:
            avg_metrics[name] = metrics[name].avg

        return avg_loss, avg_metrics

    def prepare_examples(self, batch):
        batch_size, sequence_size, time_step_size = batch.size()
        middle_tick = sequence_size // 2

        past_tensor_indices = [self.TENSOR_IDX_MAPPING[feature]
                               for feature in self.FEATURES['past']]
        past_tensor_indices += self.chord_tensor_idx
        past = self.mask_entry(
            batch[:, :middle_tick, :],
            past_tensor_indices,
            dim=2
        )

        # Remove improvised pitch and attack from present tick
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
        # Remove improvised pitch and attack from future ticks
        future_tensor_indices = [self.TENSOR_IDX_MAPPING[feature]
                                 for feature in self.FEATURES['future']]
        future_tensor_indices += self.chord_tensor_idx
        future = self.mask_entry(
            reversed_tensor,
            future_tensor_indices,
            dim=2
        )

        # Remove everything but improvised pitch and attack to get label
        label_tensor_indices = [self.TENSOR_IDX_MAPPING[feature]
                                for feature in self.LABELS]
        label = self.mask_entry(
            batch[:, middle_tick:middle_tick + 1:, :],
            label_tensor_indices,
            dim=2
        )
        label = label.view(batch_size, -1)

        return past, present, future, label

    def mask_entry(self, tensor, masked_indices, dim):
        idx = [i for i in range(tensor.size(dim)) if i in masked_indices]
        idx = Variable(torch.LongTensor(idx).cuda(), volatile=self.VOLATILE)
        tensor = tensor.index_select(dim, idx)

        return tensor

    def reverse_tensor(self, tensor, dim):
        idx = [i for i in range(tensor.size(dim) - 1, -1, -1)]
        idx = Variable(torch.LongTensor(idx).cuda(), volatile=self.VOLATILE).cuda()
        tensor = tensor.index_select(dim, idx)

        return tensor

    def loss_function(self, output_pitch, output_attack, label):
        pitch_loss = self.pitch_loss_function(output_pitch, label[:, 0])
        attack_loss = self.attack_loss_function(output_attack.float(), label[:, 1].float())

        pitch_top1, \
        pitch_top3, \
        pitch_top5 = self.accuracy(
            output_pitch,
            label[:, 0].contiguous(),
            topk=(1, 3, 5)
        )

        attack_top1, = self.accuracy(
            output_attack[:, None],
            label[:, 1].contiguous(),
            topk=(1,)
        )

        metrics = {
            'pitch_loss': pitch_loss, 'attack_loss': attack_loss,
            'pitch_top1': pitch_top1, 'pitch_top3': pitch_top3, 'pitch_top5': pitch_top5,
            'attack_top1': attack_top1
        }
        
        total_loss = self.pitch_loss_weight * pitch_loss + \
                     self.attack_loss_weight * attack_loss

        return total_loss, metrics

    @staticmethod
    def accuracy(output, label, topk=(1,)):
        maxk = max(topk)
        batch_size = label.size(0)

        _, predicted = output.topk(maxk, -1, True, True)
        predicted = predicted.t().type_as(label)
        correct = predicted.eq(label.reshape(1, -1).expand_as(predicted)).contiguous()

        results = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            results.append(correct_k.mul_(100.0 / batch_size))

        return results
