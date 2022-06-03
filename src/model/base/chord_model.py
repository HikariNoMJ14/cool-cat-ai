import os

import torch

from src.model.base import BaseModel

dir_path = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.join(dir_path, '..', '..', '..')


class ChordModel(BaseModel):

    def __init__(self, dataset=None, logger=None, save_path=os.path.join(src_path, 'results'), **kwargs):
        super(ChordModel, self).__init__(dataset, logger, save_path, **kwargs)

        self.chord_extension_count = dataset.chord_extension_count
        self.chord_encoding_type = dataset.chord_encoding_type

        self.chord_tensor_idx = list(range(
            self.TENSOR_IDX_MAPPING['chord_pitches_start'],
            self.TENSOR_IDX_MAPPING['chord_pitches_start'] + self.chord_extension_count,
        ))

    def encode_chord_pitches(self, chord_pitches):
        chord_pitches_flat = chord_pitches.view(-1)
        chord_pitches_embedding = self.pitch_encoder(chord_pitches_flat) \
            .view(chord_pitches.size(0), chord_pitches.size(1), -1)

        return chord_pitches_embedding

    # TODO refactor
    def extract_chords(self, tensor, idx):
        chord_tensor = tensor[:, :, idx[0]:idx[1] + 1]

        return torch.squeeze(chord_tensor).contiguous().view(tensor.size(0), tensor.size(1), -1)
