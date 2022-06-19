import os
import json

import numpy as np
import torch

dir_path = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.join(dir_path, '..', '..')

chord_mapping_filepath = os.path.join(src_path, 'data', 'tensor_dataset', 'chords', 'extended_7.json')


class MelodyGenerator:

    def __init__(self, model, metadata, temperature, sample, logger):

        self.model = model
        self.metadata = torch.Tensor([[metadata]]).long().cuda()
        self.sequence_size = model.sequence_size
        self.start_pitch_symbol = model.start_pitch_symbol
        self.end_pitch_symbol = model.end_pitch_symbol
        self.temperature = temperature
        self.sample = sample

        self.generated_improvised_pitches = np.array([])

        self.logger = logger

        with open(chord_mapping_filepath) as fp:
            self.chord_mapping = json.load(fp)

        self.context = None
        self.melody = None

    def seed_generation(self):
        pass

    def generate_melody(self, melody_name, n_measures):
        self.setup_context(melody_name)
        self.seed_generation()


