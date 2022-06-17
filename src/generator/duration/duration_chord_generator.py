import os

import numpy as np
import torch

from src.generator import DurationBaseGenerator
from src.melody import DurationMelody
from src.utils import get_chord_progressions
from src.utils.constants import TICKS_PER_MEASURE

dir_path = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.join(dir_path, '..', '..', '..')


class DurationChordGenerator(DurationBaseGenerator):

    def __init__(self, model, metadata, temperature, sample, logger):
        super(DurationChordGenerator, self).__init__(model, metadata, temperature, sample, logger)

    def setup_context(self, melody_name, transpose_interval=0):
        chord_progressions = get_chord_progressions(src_path)

        if melody_name not in chord_progressions:
            self.logger.error(f'Chord progression for {melody_name} not found')
            exit(1)

        chord_progression = chord_progressions[melody_name]

        self.melody = DurationMelody(None, polyphonic=False, duration_correction=0)
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
        padded_improvised_chord_pitches = []

        center_improvised_offsets = torch.from_numpy(
            self.generated_improvised_offsets[slice_start:slice_end]
        ).long().clone()[None, :].cuda()
        center_improvised_pitches = torch.from_numpy(
            self.generated_improvised_pitches[slice_start:slice_end]
        ).long().clone()[None, :].cuda()
        center_improvised_durations = torch.from_numpy(
            self.generated_improvised_durations[slice_start:slice_end]
        ).long().clone()[None, :].cuda()
        try:
            center_improvised_chord_pitches = torch.from_numpy(
                np.stack([
                    np.array(
                        self.chord_mapping[
                            self.melody.flat_chord_progression[
                                int(np.floor(
                                    tick /
                                    (TICKS_PER_MEASURE / self.melody.chord_progression_time_signature[0]))
                                ) % len(self.melody.flat_chord_progression)
                                ]
                        ])
                    for tick in self.generated_improvised_ticks[slice_start:slice_end]
                ])
            ).long().clone().transpose(0, 1).cuda()
        except ValueError:
            center_improvised_chord_pitches = torch.Tensor([]).long().cuda()

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

        padded_improvised_offsets.append(center_improvised_offsets)
        padded_improvised_pitches.append(center_improvised_pitches)
        padded_improvised_durations.append(center_improvised_durations)
        padded_improvised_chord_pitches.append(center_improvised_chord_pitches)

        padded_improvised_offsets = torch.cat(padded_improvised_offsets, 1)
        padded_improvised_pitches = torch.cat(padded_improvised_pitches, 1)
        padded_improvised_durations = torch.cat(padded_improvised_durations, 1)
        padded_improvised_chord_pitches = torch.cat(padded_improvised_chord_pitches, 1)

        past_improvised = torch.cat([
            padded_improvised_offsets[:, :middle_idx],
            padded_improvised_pitches[:, :middle_idx],
            padded_improvised_durations[:, :middle_idx],
            padded_improvised_chord_pitches[:, :middle_idx]
        ], 0).transpose(0, 1)[None, :, :].cuda()

        # First generated note
        present_offset = padded_improvised_offsets[:, middle_idx:middle_idx + 1]
        if present_offset.size(1) == 0:
            present_offset = torch.Tensor([[0]]).long().cuda()

        present_chord_pitches = padded_improvised_chord_pitches[:, middle_idx:middle_idx + 1]
        if present_chord_pitches.size(1) == 0:
            present_chord_pitches = torch.from_numpy(
                np.stack([
                    np.array(
                        self.chord_mapping[
                            self.melody.flat_chord_progression[0]
                        ])
                ])
            ).long().clone().transpose(0, 1).cuda()

        present = torch.cat([
            present_offset,
            self.metadata,
            present_chord_pitches
        ], 0).transpose(0, 1)[None, :, :].cuda()

        return past_improvised, present
