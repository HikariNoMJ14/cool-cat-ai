import os
import time

import numpy as np
import pandas as pd
import torch
from torch.functional import F

from src.generator import MelodyGenerator
from src.melody import DurationMelody
from src.utils import get_chord_progressions
from src.utils.constants import TICKS_PER_MEASURE, REST_PITCH_SYMBOL

dir_path = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.join(dir_path, '..', '..', '..')


class DurationBaseGenerator(MelodyGenerator):

    def __init__(self, model, temperature, sample, logger):
        super(DurationBaseGenerator, self).__init__(model, temperature, sample, logger)
        self.metadata = None

        self.start_duration_symbol = model.start_duration_symbol
        self.end_duration_symbol = model.end_duration_symbol

        self.generated_improvised_ticks = np.array([])
        self.generated_improvised_offsets = np.array([])
        self.generated_improvised_durations = np.array([])

    def generate_melody(self, melody_name, metadata, n_measures):
        super().generate_melody(melody_name, metadata, n_measures)

        self.generated_improvised_ticks = np.array([])
        self.generated_improvised_offsets = np.array([])
        self.generated_improvised_durations = np.array([])

        tick = 0

        with torch.no_grad():
            while tick < n_measures * TICKS_PER_MEASURE:
                generated_pitch, generated_duration = self.generate_note(tick)

                self.generated_improvised_pitches = np.append(self.generated_improvised_pitches,
                                                              generated_pitch.item())
                self.generated_improvised_durations = np.append(self.generated_improvised_durations,
                                                                generated_duration.item())

                self.generated_improvised_offsets = np.append(self.generated_improvised_offsets,
                                                              tick % TICKS_PER_MEASURE)
                self.generated_improvised_ticks = np.append(self.generated_improvised_ticks,
                                                            tick)

                tick += generated_duration.item()

    def setup_context(self, melody_name, metadata):
        self.metadata = torch.Tensor([[metadata]]).long().cuda()

        chord_progressions = get_chord_progressions(src_path)

        if melody_name not in chord_progressions:
            self.logger.error(f'Chord progression for {melody_name} not found')
            exit(1)

        chord_progression = chord_progressions[melody_name]

        self.melody = DurationMelody(None, polyphonic=False)
        self.melody.song_name = melody_name
        self.melody.set_song_structure(chord_progression)

    def get_improvised_context(self, tick):
        middle_idx = self.sequence_size // 2
        length = len(self.generated_improvised_pitches) + 1

        start_idx = length - middle_idx
        end_idx = length + middle_idx + 1

        slice_start = start_idx if start_idx > 0 else 0
        slice_end = end_idx if end_idx < length else length

        padded_improvised_offsets = []
        padded_improvised_pitches = []
        padded_improvised_durations = []

        center_improvised_offsets = torch.from_numpy(
            self.generated_improvised_offsets[slice_start:slice_end]
        ).long().clone()[None, :].cuda()
        center_improvised_pitches = torch.from_numpy(
            self.generated_improvised_pitches[slice_start:slice_end]
        ).long().clone()[None, :].cuda()
        center_improvised_durations = torch.from_numpy(
            self.generated_improvised_durations[slice_start:slice_end]
        ).long().clone()[None, :].cuda()

        if start_idx < 0:
            first_offset = int(center_improvised_offsets[:, 0]) if len(self.generated_improvised_offsets) > 0 else tick
            left_improvised_offsets = torch.from_numpy(
                np.array([np.arange(first_offset + start_idx, first_offset, 1) % TICKS_PER_MEASURE])
            ).long().clone().cuda()

            left_improvised_pitches = torch.from_numpy(
                np.array([self.start_pitch_symbol])
            ).long().clone().repeat(-start_idx, 1).transpose(0, 1).cuda()

            left_improvised_durations = torch.from_numpy(
                np.array([self.start_duration_symbol])
            ).long().clone().repeat(-start_idx, 1).transpose(0, 1).cuda()

            padded_improvised_offsets.append(left_improvised_offsets)
            padded_improvised_pitches.append(left_improvised_pitches)
            padded_improvised_durations.append(left_improvised_durations)

        padded_improvised_offsets.append(center_improvised_offsets)
        padded_improvised_pitches.append(center_improvised_pitches)
        padded_improvised_durations.append(center_improvised_durations)

        padded_improvised_offsets = torch.cat(padded_improvised_offsets, 1)
        padded_improvised_pitches = torch.cat(padded_improvised_pitches, 1)
        padded_improvised_durations = torch.cat(padded_improvised_durations, 1)

        past_improvised = torch.cat([
            padded_improvised_offsets[:, :middle_idx],
            padded_improvised_pitches[:, :middle_idx],
            padded_improvised_durations[:, :middle_idx]
        ], 0).transpose(0, 1)[None, :, :].cuda()

        # First generated note
        present_offset = padded_improvised_offsets[:, middle_idx:middle_idx + 1]
        if present_offset.size(1) == 0:
            present_offset = torch.Tensor([[0]]).long().cuda()

        present = torch.cat([
            present_offset,
            self.metadata
        ], 0).transpose(0, 1)[None, :, :].cuda()

        return past_improvised, present

    def get_context(self, tick):
        past_improvised, present = self.get_improvised_context(tick)

        assert present.eq(self.start_pitch_symbol).count_nonzero() == 0 and \
               present.eq(self.end_pitch_symbol).count_nonzero() == 0

        if self.start_pitch_symbol != self.end_pitch_symbol:
            assert past_improvised.eq(self.end_pitch_symbol).count_nonzero() == 0

        return past_improvised, present

    def generate_note(self, tick):
        past_improvised, present = self.get_context(tick)

        output_pitch, output_duration = self.model((past_improvised, present))

        output_pitch = output_pitch.squeeze()
        output_duration = output_duration.squeeze()

        pitch_probs = F.softmax(output_pitch / self.temperature, -1)
        duration_probs = F.softmax(output_duration / self.temperature, -1)

        if self.sample[0]:
            new_pitch = torch.multinomial(pitch_probs, 1)
        else:
            _, max_idx_pitch = torch.max(pitch_probs, 0)
            new_pitch = max_idx_pitch.unsqueeze(0)

        if self.sample[1]:
            new_duration = torch.multinomial(duration_probs, 1)
        else:
            _, max_idx_duration = torch.max(duration_probs, 0)
            new_duration = max_idx_duration.unsqueeze(0)

        new_duration = self.model.convert_ids_to_durations(new_duration)

        self.logger.debug([new_pitch.item(), new_duration.item()])

        assert 0 <= new_pitch <= 128
        assert new_duration > 0

        return new_pitch, new_duration

    def save(self, tempo=120, save_path=None):
        new_melody = pd.DataFrame()
        new_melody['ticks'] = pd.Series(data=self.generated_improvised_ticks)
        new_melody['offset'] = pd.Series(data=self.generated_improvised_offsets)
        new_melody['improvised_pitch'] = pd.Series(data=self.generated_improvised_pitches).replace(REST_PITCH_SYMBOL, np.nan)
        new_melody['improvised_duration'] = pd.Series(data=self.generated_improvised_durations)
        new_melody['chord_name'] = pd.Series(data=[
            self.melody.flat_chord_progression[
                int(np.floor(
                    tick /
                    (TICKS_PER_MEASURE / self.melody.chord_progression_time_signature[0]))
                ) % len(self.melody.flat_chord_progression)
                ] for tick in self.generated_improvised_ticks])

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
