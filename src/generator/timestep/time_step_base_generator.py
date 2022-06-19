import os
import time

import numpy as np
import pandas as pd
import torch
from torch.functional import F

from src.utils.constants import TICKS_PER_MEASURE, REST_PITCH_SYMBOL, REST_ATTACK_SYMBOL
from src.generator import MelodyGenerator

dir_path = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.join(dir_path, '..', '..', '..')


class TimeStepBaseGenerator(MelodyGenerator):

    def __init__(self, model, metadata, temperature, sample, logger):
        super(TimeStepBaseGenerator, self).__init__(model, metadata, temperature, sample, logger)

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

    def generate_note(self, tick):
        past_improvised, present = self.get_context(tick)

        output_pitch, output_duration = self.model((past_improvised, present))

        output_pitch = output_pitch.squeeze()
        output_duration = output_duration.squeeze()

        pitch_probs = F.softmax(output_pitch / self.temperature, -1)
        duration_probs = torch.sigmoid(output_duration)

        if self.sample[0]:
            new_pitch = torch.multinomial(pitch_probs, 1)
        else:
            _, max_idx_pitch = torch.max(pitch_probs, 0)
            new_pitch = max_idx_pitch.unsqueeze(0)

        if self.sample[1]:
            new_duration = torch.multinomial(duration_probs, 1)
        else:
            _, max_inds_duration = torch.max(duration_probs, 0)
            new_duration = max_inds_duration.unsqueeze(0)

        new_duration = self.model.convert_ids_to_durations(new_duration)

        assert 0 <= new_pitch <= 128

        if new_duration == 0:
            self.logger.error('Predicted duration is 0')
            new_duration = torch.Tensor(1)
        assert new_duration > 0

        return new_pitch, new_duration

    def save(self, save_path=None):
        new_melody = pd.DataFrame()
        ticks = range(self.generated_improvised_pitches.shape[0])

        new_melody['offset'] = pd.Series(data=list(ticks))
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

        self.melody.to_midi(out_filepath_mid, 120)  # TODO convert from metadata mapping
        self.melody.encoded.to_csv(out_filepath_csv)

        return out_filepath_csv
