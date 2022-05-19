import os
import time

import numpy as np
import pandas as pd
import torch
from torch.functional import F

from src.generator import MelodyGenerator
from src.melody import DurationMelody
from src.utils import get_chord_progressions, get_original_filepath
from src.utils.constants import TICKS_PER_MEASURE, REST_SYMBOL

dir_path = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.join(dir_path, '..', '..')


class DurationGenerator(MelodyGenerator):

    def __init__(self, model, temperature, logger):
        super(DurationGenerator, self).__init__(model, temperature, logger)

        self.start_duration_symbol = model.start_duration_symbol
        self.end_duration_symbol = model.end_duration_symbol

        self.generated_improvised_ticks = np.array([])
        self.generated_improvised_offsets = np.array([])
        self.generated_improvised_durations = np.array([])

    def generate_melody(self, melody_name, n_measures):
        super().generate_melody(melody_name, n_measures)

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

    def setup_context(self, melody_name, transpose_interval=0):
        chord_progressions = get_chord_progressions(src_path)
        original_filepath = get_original_filepath(melody_name)

        if melody_name not in chord_progressions:
            self.logger.error(f'Chord progression for {melody_name} not found')
            exit(1)

        chord_progression = chord_progressions[melody_name]

        # TODO generalize logic - also appears on duration melody
        self.melody = DurationMelody(None, polyphonic=False, duration_correction=0)
        self.melody.song_name = melody_name
        self.melody.set_song_structure(chord_progression)
        self.melody.encode(None, original_filepath)

        original_notes = self.melody.encoded[self.melody.encoded['type'] == 'original']

        # We need to add ticks to context to know what original notes belong to the past/present/future
        ticks = torch.from_numpy(
            np.array([original_notes['ticks']])
        ).long().clone()

        offsets = torch.from_numpy(
            np.array([original_notes['offset']])
        ).long().clone()

        original_pitches = torch.from_numpy(
            np.array([(original_notes['pitch'] + transpose_interval).fillna(REST_SYMBOL)])
        ).long().clone()

        original_durations = torch.from_numpy(
            np.array([original_notes['duration']])
        ).long().clone()

        chord_pitches = torch.from_numpy(
            np.stack(
                original_notes['chord_name'].apply(
                    lambda x: np.array(self.chord_mapping[x]) + transpose_interval
                )
            )
        ).long().clone().transpose(0, 1)

        self.context = torch.cat([
            ticks,
            offsets,
            original_pitches,
            original_durations,
            chord_pitches
        ], 0).transpose(0, 1)

        # TODO duplicate - similar to create_padded_tensor in model

    def get_improvised_context(self, tick):
        middle_idx = self.sequence_size // 2
        length = len(self.generated_improvised_pitches)

        start_idx = length - middle_idx
        end_idx = length + middle_idx + 1

        slice_start = start_idx if start_idx > 0 else 0
        slice_end = end_idx if end_idx < length else length

        padded_improvised_offsets = []
        padded_improvised_pitches = []
        padded_improvised_durations = []
        padded_improvised_chord_pitches = []

        if len(self.generated_improvised_pitches) > 0:
            center_improvised_offsets = torch.from_numpy(
                self.generated_improvised_offsets[slice_start:slice_end]
            ).long().clone()[None, :].cuda()
            center_improvised_pitches = torch.from_numpy(
                self.generated_improvised_pitches[slice_start:slice_end]
            ).long().clone()[None, :].cuda()
            center_improvised_durations = torch.from_numpy(
                self.generated_improvised_durations[slice_start:slice_end]
            ).long().clone()[None, :].cuda()
            center_improvised_chord_pitches = torch.from_numpy(
                np.stack([
                    np.array(
                        self.chord_mapping[
                            self.melody.flat_chord_progression[
                                int(np.floor(tick * self.melody.chord_progression_time_signature[0])) %
                                len(self.melody.flat_chord_progression)
                                ]
                        ])
                    for tick in self.generated_improvised_ticks[slice_start:slice_end]
                ])
            ).long().clone().transpose(0, 1).cuda()

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

            left_improvised_chord_pitches = torch.from_numpy(
                np.array([self.start_pitch_symbol])
            ).long().clone().repeat(-start_idx, self.model.chord_extension_count).transpose(0, 1).cuda()

            padded_improvised_offsets.append(left_improvised_offsets)
            padded_improvised_pitches.append(left_improvised_pitches)
            padded_improvised_durations.append(left_improvised_durations)
            padded_improvised_chord_pitches.append(left_improvised_chord_pitches)

        if len(self.generated_improvised_pitches) > 0:
            padded_improvised_offsets.append(center_improvised_offsets)
            padded_improvised_pitches.append(center_improvised_pitches)
            padded_improvised_durations.append(center_improvised_durations)
            padded_improvised_chord_pitches.append(center_improvised_chord_pitches)

        improvised_offsets = torch.cat(padded_improvised_offsets, 1)
        improvised_pitches = torch.cat(padded_improvised_pitches, 1)
        improvised_durations = torch.cat(padded_improvised_durations, 1)
        improvised_chord_pitches = torch.cat(padded_improvised_chord_pitches, 1)

        past_improvised = torch.cat([
            improvised_offsets[:, :middle_idx],
            improvised_pitches[:, :middle_idx],
            improvised_durations[:, :middle_idx],
            improvised_chord_pitches[:, :middle_idx]
        ], 0).transpose(0, 1)[None, :, :].cuda()

        return past_improvised

    def get_original_context(self, tick):
        middle_tick = self.sequence_size // 2
        length = self.context.size(0)

        idx = torch.nonzero(self.context[:, 0] <= tick)[-1].item()
        start_idx = idx - middle_tick
        end_idx = idx + middle_tick + 1  # TODO only works for odd numbers

        padded_original_offsets = []
        padded_original_pitches = []
        padded_original_durations = []
        padded_original_chord_pitches = []

        slice_start = start_idx if start_idx > 0 else 0
        slice_end = end_idx if end_idx < length else length

        sliced_data = self.context[slice_start:slice_end]

        # TODO remove '4'
        chord_pitches_start = 4
        chord_pitches_end = 4 + self.model.chord_extension_count

        center_original_offsets = sliced_data[None, :, 1]  # TODO remove hard-coded numbers
        center_original_pitches = sliced_data[None, :, 2]
        center_original_durations = sliced_data[None, :, 3]
        center_original_chord_pitches = sliced_data[:, chord_pitches_start:chord_pitches_end].transpose(0, 1)

        if start_idx < 0:
            first_offset = int(center_original_offsets[:, 0])

            left_original_offsets = torch.from_numpy(  # TODO check logic is correct
                np.array([np.arange(first_offset + start_idx, first_offset, 1) % TICKS_PER_MEASURE])
            ).long().clone()

            left_original_pitches = torch.from_numpy(
                np.array([self.start_pitch_symbol])
            ).long().clone().repeat(-start_idx, 1).transpose(0, 1)

            left_original_durations = torch.from_numpy(
                np.array([self.start_duration_symbol])
            ).long().clone().repeat(-start_idx, 1).transpose(0, 1)

            # TODO use chord for corresponding offset?
            left_original_chord_pitches = torch.from_numpy(
                np.array([self.start_pitch_symbol])
            ).long().clone().repeat(-start_idx, self.model.chord_extension_count).transpose(0, 1)

            padded_original_offsets.append(left_original_offsets)
            padded_original_pitches.append(left_original_pitches)
            padded_original_durations.append(left_original_durations)
            padded_original_chord_pitches.append(left_original_chord_pitches)

        padded_original_offsets.append(center_original_offsets)
        padded_original_pitches.append(center_original_pitches)
        padded_original_durations.append(center_original_durations)
        padded_original_chord_pitches.append(center_original_chord_pitches)

        if end_idx > length:
            last_offset = int(center_original_offsets[:, -1])
            right_padded_offsets = torch.from_numpy(
                np.array([np.arange(last_offset + 1, last_offset + end_idx - length + 1, 1) % TICKS_PER_MEASURE])
            ).long().clone()

            right_padding_pitches = torch.from_numpy(
                np.array([self.end_pitch_symbol])
            ).long().clone().repeat(end_idx - length, 1).transpose(0, 1)

            right_padded_durations = torch.from_numpy(
                np.array([self.end_duration_symbol])
            ).long().clone().repeat(end_idx - length, 1).transpose(0, 1)

            # TODO use chord for corresponding offset?
            right_padded_chord_pitches = torch.from_numpy(
                np.array(self.end_pitch_symbol)
            ).long().clone().repeat(end_idx - length, self.model.chord_extension_count).transpose(0, 1)

            padded_original_offsets.append(right_padded_offsets)
            padded_original_pitches.append(right_padding_pitches)
            padded_original_durations.append(right_padded_durations)
            padded_original_chord_pitches.append(right_padded_chord_pitches)

        padded_original_offsets = torch.cat(padded_original_offsets, 1)
        padded_original_pitches = torch.cat(padded_original_pitches, 1)
        padded_original_durations = torch.cat(padded_original_durations, 1)
        padded_original_chord_pitches = torch.cat(padded_original_chord_pitches, 1)

        past_original = torch.cat([
            padded_original_offsets[:, :middle_tick],
            padded_original_pitches[:, :middle_tick],
            padded_original_durations[:, :middle_tick],
            padded_original_chord_pitches[:, :middle_tick]
        ], 0).transpose(0, 1)[None, :, :].cuda()

        present = torch.cat([
            padded_original_offsets[:, middle_tick:middle_tick + 1],
            padded_original_chord_pitches[:, middle_tick:middle_tick + 1]
        ], 0).transpose(0, 1)[None, :, :].cuda()

        future = torch.cat([
            padded_original_offsets[:, middle_tick + 1:],
            padded_original_pitches[:, middle_tick + 1:],
            padded_original_durations[:, middle_tick + 1:],
            padded_original_chord_pitches[:, middle_tick + 1:]
        ], 0).transpose(0, 1)[None, :, :].cuda()
        future = self.reverse_tensor(future, dim=0)

        return past_original, present, future

    def get_context(self, tick):
        past_improvised = self.get_improvised_context(tick)
        past_original, present, future = self.get_original_context(tick)

        assert past_improvised.eq(self.end_pitch_symbol).count_nonzero() == 0
        assert past_original.eq(self.end_pitch_symbol).count_nonzero() == 0
        assert present.eq(self.start_pitch_symbol).count_nonzero() == 0 and \
               present.eq(self.end_pitch_symbol).count_nonzero() == 0
        assert future.eq(self.start_pitch_symbol).count_nonzero() == 0

        return past_improvised, past_original, present, future

    def generate_note(self, tick):
        past_improvised, past_original, present, future = self.get_context(tick)

        output_pitch, output_duration = self.model((past_improvised, past_original, present, future))

        output_pitch = output_pitch.squeeze()
        output_duration = output_duration.squeeze()

        pitch_probs = F.softmax(output_pitch / self.temperature, -1)
        duration_probs = torch.sigmoid(output_duration)

        stochastic_search = True
        top_p = True
        p = 0.9

        if stochastic_search:
            _, max_inds_pitch = torch.max(pitch_probs, 0)
            _, max_inds_duration = torch.max(duration_probs, 0)

            new_pitch = max_inds_pitch.unsqueeze(0)
            new_duration = max_inds_duration.unsqueeze(0)
        elif top_p:
            topp_p = self.mask_non_top_p(p, pitch_probs)
            topp_d = self.mask_non_top_p(p, duration_probs)

            new_pitch = torch.distributions.categorical.Categorical(probs=topp_p).sample().unsqueeze(-1)
            new_duration = torch.distributions.categorical.Categorical(probs=topp_d).sample().unsqueeze(-1)
        else:
            new_pitch = torch.multinomial(pitch_probs, 1)
            new_duration = torch.multinomial(duration_probs, 1)

        new_duration = self.model.convert_ids_to_durations(new_duration)

        assert 0 <= new_pitch <= 128
        assert new_duration > 0

        return new_pitch, new_duration

    def save(self):
        new_melody = pd.DataFrame()
        new_melody['ticks'] = pd.Series(data=self.generated_improvised_ticks)
        new_melody['offset'] = pd.Series(data=self.generated_improvised_offsets)
        new_melody['improvised_pitch'] = pd.Series(data=self.generated_improvised_pitches).replace(REST_SYMBOL, np.nan)
        new_melody['improvised_duration'] = pd.Series(data=self.generated_improvised_durations)

        self.melody.encoded = new_melody

        out_path = os.path.join(
            src_path,
            'data', 'generated',
            self.model.name
        )

        if not os.path.exists(out_path):
            os.makedirs(out_path)

        filename = f'{time.strftime("%y_%m_%d_%H_%M_%S")} {self.melody.song_name}.mid'
        self.melody.to_midi(os.path.join(out_path, filename))
