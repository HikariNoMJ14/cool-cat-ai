import torch
import torch.nn as nn
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class MonoTimeStepModel(nn.Module):
    # Input parameters
    offset_size = 48
    pitch_size = 129
    attack_size = 1
    metadata_size = 0

    # embedding parameters
    embedding_size = 16  # TODO add metadata embedding?

    # LSTM parameters
    num_layers = 3
    lstm_hidden_size = 512

    # NN parameters
    present_nn_output_size = 32

    # Train parameters
    num_epochs = 10
    batch_size = 100

    def __init__(self, chord_encoding_type, chord_extension_count):  # TODO add dropout
        super(MonoTimeStepModel, self).__init__()

        self.offset_encoder = nn.Embedding(self.offset_size, self.embedding_size)
        self.pitch_encoder = nn.Embedding(self.pitch_size, self.embedding_size, scale_grad_by_freq=True) # TODO add padding_idx for 128 (rest)?

        #  offset +
        #  improvised_pitch + improvised_attack +
        #  original_pitch + original_attack +
        #  chord_pitch * number_of_pitches
        lstm_input_size = self.embedding_size + \
                          self.embedding_size + self.attack_size + \
                          self.embedding_size + self.attack_size + \
                          self.embedding_size * chord_extension_count if chord_encoding_type != 'compressed' else 12

        self.past_lstm_layer_1 = nn.LSTM(lstm_input_size)
        self.past_lstm_layer_2 = nn.LSTM(self.lstm_hidden_size)
        self.past_lstm_layer_3 = nn.LSTM(self.lstm_hidden_size)

        self.past_lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.num_layers,
            dropout=None,
            batch_first=True
        )

        present_nn_input_size = self.embedding_size + self.attack_size + \
                                self.embedding_size + self.attack_size + \
                                self.metadata_size

        self.present_nn = nn.Linear(present_nn_input_size, self.present_nn_output_size)

        self.future_lstm_layer_1 = nn.LSTM(lstm_input_size)
        self.future_lstm_layer_2 = nn.LSTM(self.lstm_hidden_size)
        self.future_lstm_layer_3 = nn.LSTM(self.lstm_hidden_size)

        self.future_lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.num_layers,
            dropout=None,
            batch_first=True
        )

        merge_nn_input_size = self.lstm_hidden_size + self.present_nn_output_size + self.lstm_hidden_size
        merge_nn_output_size = self.embedding_size + self.attack_size
        self.merge_nn = nn.Linear(merge_nn_input_size, merge_nn_output_size)

        self.pitch_decoder = nn.Linear(self.embedding_size, self.pitch_size)

        # tie pitch econder and decoder weights
        self.pitch_decoder.weight = self.pitch_encoder.weight

    # From DeepBach
    def init_hidden(self):
        hidden = (
            torch.randn(self.num_layers, self.batch_size, self.lstm_hidden_size).cuda(),
            torch.randn(self.num_layers, self.batch_size, self.lstm_hidden_size).cuda()
        )
        return hidden

    # From BebopNet
    # def init_hidden(self, batch_size):
    #     weight = next(self.parameters()).detach()
    #     return (weight.new(self.num_layers, batch_size, self.hidden_size).zero_(),
    #             weight.new(self.num_layers, batch_size, self.hidden_size).zero_())

    def forward(self, inputs, hidden_past, hidden_future):
        sequence_length = inputs.size()[0]
        batch_size = inputs.size()[1]

        offset = None
        improvised_pitches = None
        improvised_attacks = None
        original_pitches = None
        original_attacks = None
        chord_bass = None
        chord_root = None
        chord_pitches = None

        offset_embedding = self.offset_encoder(offset)

        improvised_pitch_embedding = self.pitch_encoder(improvised_pitches)
        original_pitch_embedding = self.pitch_encoder(original_pitches)

        chord_root_embedding = self.pitch_encoder(chord_root)
        chord_bass_embedding = self.pitch_encoder(chord_bass)
        chord_pitches_embedding = self.pitch_encoder(chord_pitches.view(-1))

        lstm_input = torch.cat([
            offset_embedding,
            improvised_pitch_embedding, improvised_attacks,
            original_pitch_embedding, original_attacks,
            chord_root_embedding, chord_bass_embedding, chord_pitches_embedding
        ])

        # TODO add dropout(s) ?

        # Past LSTM

        past_lstm_input = None
        past_lstm_hidden = self.init_hidden()
        past_lstm_output, past_lstm_hidden = self.past_lstm(past_lstm_input, past_lstm_hidden)

        past_lstm_output = past_lstm_output[:, -1, :]

        # Present NN

        present_nn_output = self.present_nn()

        # Future LSTM

        future_lstm_input = None
        future_lstm_hidden = self.init_hidden()
        future_lstm_output, future_lstm_hidden = self.future_lstm(future_lstm_input, future_lstm_hidden)

        future_lstm_output = future_lstm_output[:, -1, :]

        merge_nn_input = torch.cat([past_lstm_output, present_nn_output, future_lstm_output], 1)

    def train(self):
        logger.info(f'--- Training ---')

        for epoch in range(self.num_epochs):
            logger.info(f'--- Epoch {epoch} ---')
