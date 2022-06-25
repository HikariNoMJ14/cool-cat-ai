import os

import numpy as np
import torch

from src.melody import TimeStepMelody
from src.generator import TimeStepBaseGenerator
from src.utils import get_chord_progressions

dir_path = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.join(dir_path, '..', '..', '..')


class TimeStepChordGenerator(TimeStepBaseGenerator):

    def __init__(self, model, temperature, sample, logger):
        super(TimeStepChordGenerator, self).__init__(model, temperature, sample, logger)

    def setup_context(self, melody_name, metadata, transpose_interval=0):
        self.metadata = torch.Tensor([[metadata]]).long().cuda()

        chord_progressions = get_chord_progressions(src_path)

        if melody_name not in chord_progressions:
            self.logger.error(f'Chord progression for {melody_name} not found')
            exit(1)

        chord_progression = chord_progressions[melody_name]

        self.melody = TimeStepMelody(None, polyphonic=False)
        self.melody.song_name = melody_name
        self.melody.set_song_structure(chord_progression)

        offsets = torch.from_numpy(
            np.array([self.melody.encoded['offset']])
        ).long().clone()

        chord_pitches = torch.from_numpy(
            np.stack(
                self.melody.encoded['chord_name'].apply(
                    lambda x: np.array(self.chord_mapping[x]) + transpose_interval
                )
            )
        ).long().clone().transpose(0, 1)

        self.context = torch.cat([
            offsets,
            chord_pitches
        ], 0).transpose(0, 1)

    def get_context(self, tick):
        middle_tick = self.sequence_size // 2
        start_tick = tick - middle_tick
        end_tick = tick + middle_tick + 1  # TODO only works for odd numbers
        length = self.context.size(0)

        common_sliced_data = self.context[np.arange(start_tick, end_tick) % length]

        offsets = common_sliced_data[None, :, 0].cuda()

        chord_pitches = torch.from_numpy(
            np.stack(
                common_sliced_data[:, 3:10]
            )
        ).long().clone().transpose(0, 1).cuda()

        padded_improvised_pitches = []
        padded_improvised_attacks = []

        if start_tick < 0:
            left_improvised_pitches = torch.from_numpy(
                np.array([self.start_pitch_symbol])
            ).long().clone().repeat(-start_tick, 1).transpose(0, 1).cuda()

            left_improvised_attacks = torch.from_numpy(
                np.array([self.start_attack_symbol])
            ).long().clone().repeat(-start_tick, 1).transpose(0, 1).cuda()

            padded_improvised_pitches.append(left_improvised_pitches)
            padded_improvised_attacks.append(left_improvised_attacks)

        center_improvised_pitches = torch.from_numpy(
            self.generated_improvised_pitches[-middle_tick:]
        ).long().clone()[None, :].cuda()
        center_improvised_attacks = torch.from_numpy(
            self.generated_improvised_attacks[-middle_tick:]
        ).long().clone()[None, :].cuda()

        padded_improvised_pitches.append(center_improvised_pitches)
        padded_improvised_attacks.append(center_improvised_attacks)

        improvised_pitches = torch.cat(padded_improvised_pitches, 1)
        improvised_attacks = torch.cat(padded_improvised_attacks, 1)

        past = torch.cat([
            offsets[:, :middle_tick],
            improvised_pitches[:, :middle_tick],
            improvised_attacks[:, :middle_tick],
            chord_pitches[:, :middle_tick]
        ], 0).transpose(0, 1)[None, :, :].cuda()

        present = torch.cat([
            offsets[:, middle_tick:middle_tick + 1],
            self.metadata,
            chord_pitches[:, middle_tick:middle_tick + 1]
        ], 0).transpose(0, 1)[None, :, :].cuda()

        assert present.eq(self.start_pitch_symbol).count_nonzero() == 0 and present.eq(
            self.end_pitch_symbol).count_nonzero() == 0

        if self.start_pitch_symbol != self.end_pitch_symbol:
            assert past.eq(self.end_pitch_symbol).count_nonzero() == 0

        return past, present
