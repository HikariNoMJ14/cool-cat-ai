import os
import time

import numpy as np
import pandas as pd
import torch
from torch.functional import F

from src.melody import TimeStepMelody
from src.utils import get_chord_progressions, get_original_filepath
from src.utils.constants import TICKS_PER_MEASURE, REST_PITCH_SYMBOL, REST_ATTACK_SYMBOL
from src.generator import MelodyGenerator

dir_path = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.join(dir_path, '..', '..', '..')


class TimeStepBaseGenerator(MelodyGenerator):

    def __init__(self, model, temperature, sample, logger):
        super(TimeStepBaseGenerator, self).__init__(model, temperature, sample, logger)
        self.metadata = None

        self.start_attack_symbol = model.start_attack_symbol
        self.end_attack_symbol = model.end_attack_symbol

        self.generated_improvised_attacks = np.array([])

    def generate_melody(self, melody_name, metadata, n_measures):
        super().generate_melody(melody_name, metadata, n_measures)

        with torch.no_grad():
            for measure in range(n_measures):
                for offset in range(TICKS_PER_MEASURE):
                    tick = measure * TICKS_PER_MEASURE + offset
                    generated_pitch, generated_attack = self.generate_note(tick)

                    self.generated_improvised_pitches = np.append(self.generated_improvised_pitches,
                                                                  generated_pitch.item())
                    self.generated_improvised_attacks = np.append(self.generated_improvised_attacks,
                                                                  generated_attack.item())

    def setup_context(self, melody_name, metadata):
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

        self.context = torch.cat([
            offsets
        ], 0).transpose(0, 1)

    def get_context(self, tick):
        middle_tick = self.sequence_size // 2
        start_tick = tick - middle_tick
        end_tick = tick + middle_tick + 1  # TODO only works for odd numbers
        length = self.context.size(0)

        common_sliced_data = self.context[np.arange(start_tick, end_tick) % length]

        offsets = common_sliced_data[None, :, 0].cuda()

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
            improvised_attacks[:, :middle_tick]
        ], 0).transpose(0, 1)[None, :, :].cuda()

        present = torch.cat([
            offsets[:, middle_tick:middle_tick + 1],
            self.metadata
        ], 0).transpose(0, 1)[None, :, :].cuda()

        assert present.eq(self.start_pitch_symbol).count_nonzero() == 0 and present.eq(
            self.end_pitch_symbol).count_nonzero() == 0

        if self.start_pitch_symbol != self.end_pitch_symbol:
            assert past.eq(self.end_pitch_symbol).count_nonzero() == 0

        return past, present

    def generate_note(self, tick):
        past, present = self.get_context(tick)

        output_pitch, output_attack = self.model((past, present))

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

    def save(self, tempo=120, save_path=None):
        new_melody = pd.DataFrame()
        ticks = np.array(range(self.generated_improvised_pitches.shape[0]))
        offsets = np.array(range(self.generated_improvised_pitches.shape[0])) % TICKS_PER_MEASURE

        new_melody['offset'] = pd.Series(data=offsets)
        new_melody['improvised_pitch'] = pd.Series(data=self.generated_improvised_pitches).replace(
            REST_PITCH_SYMBOL, np.nan)
        new_melody['improvised_attack'] = pd.Series(data=self.generated_improvised_attacks).replace(
            REST_ATTACK_SYMBOL, np.nan)
        new_melody['chord_name'] = pd.Series(data=[
            self.melody.flat_chord_progression[
                int(np.floor(
                    tick /
                    (TICKS_PER_MEASURE / self.melody.chord_progression_time_signature[0]))
                ) % len(self.melody.flat_chord_progression)
                ] for tick in ticks])

        self.melody.encoded = new_melody

        out_path = os.path.join(
            src_path,
            'data', 'generated',
            self.model.name
        )

        if save_path is not None:
            out_path = os.path.join(out_path, save_path)

        if not os.path.exists(out_path):
            os.makedirs(out_path)

        filename = f'{time.strftime("%y_%m_%d_%H_%M_%S")} {self.melody.song_name}'
        filename_mid = f'{filename}.mid'
        filename_csv = f'{filename}.csv'
        out_filepath_mid = os.path.join(out_path, filename_mid)
        out_filepath_csv = os.path.join(out_path, filename_csv)

        self.melody.to_midi(out_filepath_mid, tempo)
        self.melody.encoded.to_csv(out_filepath_csv)

        return out_filepath_csv
