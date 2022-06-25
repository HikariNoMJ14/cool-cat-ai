import os

import numpy as np
import torch
from torch.functional import F

from src.generator import DurationChordGenerator
from src.melody import DurationMelody
from src.utils import get_chord_progressions, get_original_filepath, reverse_tensor
from src.utils.constants import TICKS_PER_MEASURE, REST_PITCH_SYMBOL

dir_path = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.join(dir_path, '..', '..', '..')


class DurationFullGenerator(DurationChordGenerator):

    def __init__(self, model, temperature, sample, logger):
        super(DurationFullGenerator, self).__init__(model, temperature, sample, logger)

        self.start_duration_symbol = model.start_duration_symbol
        self.end_duration_symbol = model.end_duration_symbol

        self.generated_improvised_ticks = np.array([])
        self.generated_improvised_offsets = np.array([])
        self.generated_improvised_durations = np.array([])

    def setup_context(self, melody_name, metadata, transpose_interval=0):
        self.metadata = torch.Tensor([[metadata]]).long().cuda()

        chord_progressions = get_chord_progressions(src_path)
        original_filepath = get_original_filepath(melody_name)

        if melody_name not in chord_progressions:
            self.logger.error(f'Chord progression for {melody_name} not found')
            exit(1)

        chord_progression = chord_progressions[melody_name]

        self.melody = DurationMelody(None, polyphonic=False)
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
            np.array([(original_notes['pitch'] + transpose_interval).fillna(REST_PITCH_SYMBOL)])
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

        future = torch.cat([
            padded_original_offsets[:, middle_tick + 1:],
            padded_original_pitches[:, middle_tick + 1:],
            padded_original_durations[:, middle_tick + 1:],
            padded_original_chord_pitches[:, middle_tick + 1:]
        ], 0).transpose(0, 1)[None, :, :].cuda()
        future = reverse_tensor(future, dim=0)

        return past_original, future

    def get_context(self, tick):
        past_improvised, present = self.get_improvised_context(tick)
        past_original, future = self.get_original_context(tick)

        assert present.eq(self.start_pitch_symbol).count_nonzero() == 0 and \
               present.eq(self.end_pitch_symbol).count_nonzero() == 0

        if self.start_pitch_symbol != self.end_pitch_symbol:
            assert past_improvised.eq(self.end_pitch_symbol).count_nonzero() == 0
            assert past_original.eq(self.end_pitch_symbol).count_nonzero() == 0
            assert future.eq(self.start_pitch_symbol).count_nonzero() == 0

        return past_improvised, past_original, present, future

    def generate_note(self, tick):
        past_improvised, past_original, present, future = self.get_context(tick)

        output_pitch, output_duration = self.model((past_improvised, past_original, present, future))

        output_pitch = output_pitch.squeeze()
        output_duration = output_duration.squeeze()

        pitch_probs = F.softmax(output_pitch / self.temperature, -1)
        duration_probs = F.softmax(output_duration / self.temperature, -1)  # TODO check this works

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
