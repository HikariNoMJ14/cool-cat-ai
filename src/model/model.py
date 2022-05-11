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

from src.utils import Metric

dir_path = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.join(dir_path, '..', '..')


class Model(nn.Module):

    VOLATILE = False
    LOG_INTERVAL = 500

    def __init__(self, dataset=None, logger=None, save_path=os.path.join(src_path, 'results'), **kwargs):
        super(Model, self).__init__()

        self.name = ''
        self.save_dir = os.path.join(src_path, 'results')

        self.dataset = dataset
        self.logger = logger
        self.save_path = save_path

        self.logger.info('--- Init Model ---')

        self.logger.info(f'Using dataset: {dataset.name}')

        self.start_symbol = kwargs['start_symbol']
        self.end_symbol = kwargs['end_symbol']

        # Set model parameters
        self.offset_size = kwargs['offset_size']
        self.pitch_size = kwargs['pitch_size']
        self.metadata_size = kwargs['metadata_size']

        self.embedding_size = kwargs['embedding_size']
        self.lstm_num_layers = kwargs['lstm_num_layers']
        self.lstm_hidden_size = kwargs['lstm_hidden_size']
        self.nn_hidden_size = kwargs['nn_hidden_size']
        self.nn_output_size = kwargs['nn_output_size']

        self.embedding_dropout_rate = kwargs['embedding_dropout_rate']
        self.lstm_dropout_rate = kwargs['lstm_dropout_rate']
        self.nn_dropout_rate = kwargs['nn_dropout_rate']

        self.normalize = kwargs['normalize']
        self.gradient_clipping = kwargs['gradient_clipping']

        self.pitch_loss_weight = kwargs['pitch_loss_weight']

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

        self.pitch_decoder = nn.Linear(self.embedding_size, self.pitch_size)

        # Tie pitch encoder and decoder weights
        self.pitch_decoder.weight = self.pitch_encoder[0].weight

    def load(self, model_path):
        self.load_state_dict(torch.load(open(model_path, 'rb')))

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

    # TODO refactor
    def extract_features(self, tensor, feature_name, idx):
        feature_tensor = tensor[:, :, idx]

        return torch.squeeze(feature_tensor).contiguous().view(tensor.size(0), -1)

    # TODO refactor
    def extract_chords(self, tensor, idx):
        chord_tensor = tensor[:, :, idx[0]:idx[1] + 1]

        return torch.squeeze(chord_tensor).contiguous().view(tensor.size(0), tensor.size(1), -1)

    def train_and_eval(self, num_batches, batch_size, num_epochs, optimizer, scheduler, seed, callback):
        self.name = f'{self.dataset.name}_batchsize_{batch_size}_seed_{seed}'

        checkpoint_path = os.path.join(self.save_path, self.name + '.pt')
        log_path = os.path.join(self.save_path, 'log.log')
        results_path = os.path.join(self.save_path, 'results.csv')

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(self.logger.handlers[0].formatter)
        self.logger.addHandler(file_handler)

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

                train_dataset, \
                val_dataset, \
                test_dataset = self.dataset.split(
                    split=(.85, .15, 0),
                    seed=seed
                )

                self.logger.info(f'--- Epoch {epoch} ---')

                train_loss, train_metrics = self.loss_and_accuracy(
                    dataset=train_dataset,
                    optimizer=optimizer,
                    phase='train',
                    batch_size=batch_size,
                    num_batches=num_batches
                )

                valid_loss, valid_metrics = self.loss_and_accuracy(
                    dataset=val_dataset,
                    optimizer=optimizer,
                    phase='test',
                    batch_size=batch_size,
                    num_batches=int(num_batches // 5)
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

            self.plot_metrics(training_results)

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
                self.plot_metrics(training_results)

                with open(checkpoint_path, 'wb') as f:
                    torch.save(self, f)

    def loss_and_accuracy(self, dataset, phase, optimizer, batch_size, num_batches=None):
        if phase == 'train':
            self.train()
        elif phase == 'eval' or phase == 'test':
            self.eval()

        loss = Metric()
        metrics = {k: Metric() for k in self.METRICS_LIST}
        logging_loss = Metric()
        logging_metrics = {k: Metric() for k in self.METRICS_LIST}
        start_time = time.time()

        if num_batches is None:
            num_batches = self.get_num_batches(dataset, batch_size)

        for i in range(num_batches):
            batch = self.get_batch(dataset, batch_size)

            features, label = self.prepare_examples(batch)
            prediction = self(features)
            current_loss, current_metrics = self.loss_function(prediction, label)

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

    def normalize_embeddings(self):
        self.pitch_decoder.weight.data = F.normalize(self.pitch_encoder.weight, p=2, dim=1)

    def get_num_batches(self, dataset, batch_size):
        total_len = 0
        for example in dataset:
            total_len += example.size(0)

        return int(total_len // batch_size)

    def get_batch(self, dataset, batch_size):
        batch = []

        for i in range(batch_size):
            random_example_idx = np.random.randint(0, len(dataset))
            chosen_example = dataset[random_example_idx]

            mid_point = self.sequence_size // 2
            random_idx = np.random.randint(-mid_point, len(chosen_example) - mid_point)

            padded_example = self.create_padded_tensor(chosen_example, random_idx)[None, :, :]
            batch.append(padded_example)

        return torch.cat(batch, 0)

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

    def plot_metrics(self, training_results):
        for metric in self.PLOTTED_METRICS:
            fig, ax = plt.subplots(nrows=1, ncols=1)
            for phase in ['train', 'valid']:
                ax.plot(training_results[f'{phase}_{metric}'], label=f'{phase}_{metric.replace("_", " ")}')
            fig.legend(loc="upper right")
            fig.savefig(os.path.join(self.save_path, f'{metric}.png'))
            plt.close(fig)

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
