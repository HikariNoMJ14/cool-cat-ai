import os

import numpy as np
import torch
from torch.functional import F

from src.generator import TimeStepChordGenerator
from src.melody import TimeStepMelody
from src.utils import get_chord_progressions, get_original_filepath, reverse_tensor
from src.utils.constants import REST_PITCH_SYMBOL, REST_ATTACK_SYMBOL

dir_path = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.join(dir_path, '..', '..', '..')


class TimeStepFullGenerator(TimeStepChordGenerator):

    def __init__(self, model, temperature, sample, logger):
        super(TimeStepFullGenerator, self).__init__(model, temperature, sample, logger)

        self.start_attack_symbol = model.start_attack_symbol
        self.end_attack_symbol = model.end_attack_symbol

        self.generated_improvised_attacks = np.array([])

    def setup_context(self, melody_name, metadata, transpose_interval=0):
        self.metadata = torch.Tensor([[metadata]]).long().cuda()

        chord_progressions = get_chord_progressions(src_path)
        original_filepath = get_original_filepath(melody_name)

        if melody_name not in chord_progressions:
            self.logger.error(f'Chord progression for {melody_name} not found')
            exit(1)

        chord_progression = chord_progressions[melody_name]

        self.melody = TimeStepMelody(None, polyphonic=False, duration_correction=0)
        self.melody.song_name = melody_name
        self.melody.set_song_structure(chord_progression)
        self.melody.encode(None, original_filepath)

        offsets = torch.from_numpy(
            np.array([self.melody.encoded['offset']])
        ).long().clone()

        original_pitches = torch.from_numpy(
            np.array([(self.melody.encoded['original_pitch'] + transpose_interval).fillna(REST_PITCH_SYMBOL)])
        ).long().clone()

        original_attacks = torch.from_numpy(
            np.array([self.melody.encoded['original_attack'].fillna(REST_ATTACK_SYMBOL)])
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
            original_pitches,
            original_attacks,
            chord_pitches
        ], 0).transpose(0, 1)

    def get_context(self, tick):
        start_tick = tick
        end_tick = tick + self.sequence_size
        length = self.context.size(0)
        middle_tick = self.sequence_size // 2

        common_sliced_data = self.context[np.arange(start_tick, end_tick) % length]

        offsets = common_sliced_data[None, :, 0].cuda()
        original_pitches = common_sliced_data[None, :, 1].cuda()
        original_attacks = common_sliced_data[None, :, 2].cuda()

        chord_pitches = torch.from_numpy(
            np.stack(
                common_sliced_data[:, 3:11]
            )
        ).long().clone().transpose(0, 1).cuda()

        padded_improvised_pitches = []
        padded_improvised_attacks = []

        if tick < middle_tick:
            left_improvised_pitches = torch.from_numpy(
                np.array([self.start_pitch_symbol])
            ).long().clone().repeat(middle_tick - start_tick, 1).transpose(0, 1).cuda()

            left_improvised_attacks = torch.from_numpy(
                np.array([self.start_attack_symbol])
            ).long().clone().repeat(middle_tick - start_tick, 1).transpose(0, 1).cuda()

            padded_improvised_pitches.append(left_improvised_pitches)
            padded_improvised_attacks.append(left_improvised_attacks)

        center_improvised_pitches = torch.from_numpy(
            self.generated_improvised_pitches
        ).long().clone()[None, :].cuda()
        center_improvised_attacks = torch.from_numpy(
            self.generated_improvised_attacks
        ).long().clone()[None, :].cuda()

        padded_improvised_pitches.append(center_improvised_pitches)
        padded_improvised_attacks.append(center_improvised_attacks)

        if tick > middle_tick:
            right_improvised_pitches = torch.from_numpy(
                np.array([self.end_pitch_symbol])
            ).long().clone().repeat(end_tick - middle_tick, 1).transpose(0, 1).cuda()

            right_improvised_attacks = torch.from_numpy(
                np.array([self.end_attack_symbol])
            ).long().clone().repeat(end_tick - middle_tick, 1).transpose(0, 1).cuda()

            padded_improvised_pitches.append(right_improvised_pitches)
            padded_improvised_attacks.append(right_improvised_attacks)

        improvised_pitches = torch.cat(padded_improvised_pitches, 1)
        improvised_attacks = torch.cat(padded_improvised_attacks, 1)

        past = torch.cat([
            offsets[:, :middle_tick],
            improvised_pitches[:, :middle_tick],
            improvised_attacks[:, :middle_tick],
            original_pitches[:, :middle_tick],
            original_attacks[:, :middle_tick],
            chord_pitches[:, :middle_tick]
        ], 0).transpose(0, 1)[None, :, :].cuda()

        present = torch.cat([
            offsets[:, middle_tick:middle_tick + 1],
            original_pitches[:, middle_tick:middle_tick + 1],
            original_attacks[:, middle_tick:middle_tick + 1],
            self.metadata,
            chord_pitches[:, middle_tick:middle_tick + 1]
        ], 0).transpose(0, 1)[None, :, :].cuda()

        future = torch.cat([
            offsets[:, middle_tick + 1:],
            original_pitches[:, middle_tick + 1:],
            original_attacks[:, middle_tick + 1:],
            chord_pitches[:, middle_tick + 1:]
        ], 0).transpose(0, 1)[None, :, :].cuda()
        future = reverse_tensor(future, dim=0)

        assert present.eq(self.start_pitch_symbol).count_nonzero() == 0 and present.eq(
            self.end_pitch_symbol).count_nonzero() == 0

        if self.start_pitch_symbol != self.end_pitch_symbol:
            assert past.eq(self.end_pitch_symbol).count_nonzero() == 0
            assert future.eq(self.start_pitch_symbol).count_nonzero() == 0

        return past, present, future

    def generate_note(self, tick):
        past, present, future = self.get_context(tick)

        output_pitch, output_attack = self.model((past, present, future))

        output_pitch = output_pitch.squeeze()
        output_attack = output_attack.squeeze()

        pitch_probs = F.softmax(output_pitch / self.temperature, -1)
        attack_probs = F.softmax(output_attack / self.temperature, -1)

        if self.sample[0]:
            new_pitch = torch.multinomial(pitch_probs, 1)
        else:
            _, max_idx_pitch = torch.max(pitch_probs, 0)
            new_pitch = max_idx_pitch.unsqueeze(0)

        if self.sample[1]:
            new_attack = torch.multinomial(attack_probs, 1)
        else:
            _, max_idx_attack = torch.max(attack_probs, 0)
            new_attack = max_idx_attack.unsqueeze(0)

        self.logger.debug([new_pitch.item(), new_attack.item()])

        assert 0 <= new_pitch <= 128
        assert 0 <= new_attack <= 2
        assert new_pitch == 128 or new_attack != 2

        return new_pitch, new_attack
