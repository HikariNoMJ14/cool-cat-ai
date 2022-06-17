import os
import time

import numpy as np
import pandas as pd
import torch
from torch.functional import F

from src.generator import MelodyGenerator
from src.melody import TimeStepMelody
from src.utils import get_chord_progressions, get_original_filepath, reverse_tensor
from src.utils.constants import TICKS_PER_MEASURE, REST_PITCH_SYMBOL, REST_ATTACK_SYMBOL

dir_path = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.join(dir_path, '..', '..')


class TimeStepGenerator(MelodyGenerator):

    def __init__(self, model, temperature, sample, logger):
        super(TimeStepGenerator, self).__init__(model, temperature, sample, logger)

        self.start_attack_symbol = model.start_attack_symbol
        self.end_attack_symbol = model.end_attack_symbol

        self.generated_improvised_attacks = np.array([])

    def generate_melody(self, melody_name, n_measures):
        super().generate_melody(melody_name, n_measures)

        with torch.no_grad():
            for measure in range(n_measures):
                for offset in range(TICKS_PER_MEASURE):
                    tick = measure * TICKS_PER_MEASURE + offset
                    generated_pitch, generated_attack = self.generate_note(tick)

                    self.generated_improvised_pitches = np.append(self.generated_improvised_pitches,
                                                                  generated_pitch.item())
                    self.generated_improvised_attacks = np.append(self.generated_improvised_attacks,
                                                                  generated_attack.item())

    def setup_context(self, melody_name, transpose_interval=0):
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

    # TODO check whether padding logic is correct
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
                common_sliced_data[:, 3:10]
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
            chord_pitches[:, middle_tick:middle_tick + 1]
        ], 0).transpose(0, 1)[None, :, :].cuda()

        future = torch.cat([
            offsets[:, middle_tick + 1:],
            original_pitches[:, middle_tick + 1:],
            original_attacks[:, middle_tick + 1:],
            chord_pitches[:, middle_tick + 1:]
        ], 0).transpose(0, 1)[None, :, :].cuda()
        future = reverse_tensor(future, dim=0)

        assert past.eq(self.end_pitch_symbol).count_nonzero() == 0
        assert present.eq(self.start_pitch_symbol).count_nonzero() == 0 and present.eq(
            self.end_pitch_symbol).count_nonzero() == 0
        assert future.eq(self.start_pitch_symbol).count_nonzero() == 0

        return past, present, future

    def generate_note(self, tick):
        past, present, future = self.get_context(tick)

        output_pitch, output_attack = self.model(past, present, future)

        output_pitch = output_pitch.squeeze()
        output_attack = output_attack.squeeze()

        pitch_probs = F.softmax(output_pitch / self.temperature, -1)
        attack_probs = torch.sigmoid(output_attack)

        stochastic_search = True
        top_p = True
        p = 0.9

        if stochastic_search:
            _, max_inds_pitch = torch.max(pitch_probs, 0)

            new_pitch = max_inds_pitch.unsqueeze(0)
            new_attack = attack_probs.round().unsqueeze(0)
        elif top_p:
            topp_p = self.mask_non_top_p(p, pitch_probs)

            new_pitch = torch.distributions.categorical.Categorical(probs=topp_p).sample().unsqueeze(-1)
        else:
            new_pitch = torch.multinomial(pitch_probs, 1)

        self.logger.info([new_pitch.item(), new_attack.item()])

        assert 0 <= new_pitch <= 128
        assert new_attack == 0 or new_attack == 1

        if new_pitch == 128 and new_attack == 1:
            self.logger.error('Attack on rest!!!')
            new_attack = torch.Tensor([0])  # TODO remove - not ok

        return new_pitch, new_attack

    def save(self):
        self.melody.encoded['improvised_pitch'] = pd.Series(data=self.generated_improvised_pitches).replace(
            REST_PITCH_SYMBOL,
            np.nan)
        self.melody.encoded['improvised_attack'] = pd.Series(data=self.generated_improvised_attacks).replace(
            REST_ATTACK_SYMBOL,
            np.nan)

        out_path = os.path.join(
            src_path,
            'data', 'generated',
            self.model.name
        )

        if not os.path.exists(out_path):
            os.makedirs(out_path)

        filename = f'{time.strftime("%y_%m_%d_%H_%M_%S")} {self.melody.song_name}.mid'
        self.melody.to_midi(os.path.join(out_path, filename))
