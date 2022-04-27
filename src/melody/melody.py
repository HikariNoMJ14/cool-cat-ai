import os

import music21
from music21 import converter

import mido
from mido import MidiFile, MidiTrack

from mingus.core import chords
import mingus.core.notes as notes

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.ezchord import Chord
from src.utils import is_weakly_polyphonic, is_strongly_polyphonic, \
    notes_to_midi, notes_and_chord_to_midi, flatten_chord_progression, filepath_to_song_name
from src.objective_metrics import replace_enharmonic

dir_path = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.join(dir_path, '..', '..')


def no_errors(func):
    def inner(*args):
        if len(args[0].errors) == 0:
            func(*args)
        else:
            print(args[0].song_name, args[0].errors)

    return inner


class Melody:
    FINAL_TICKS_PER_BEAT = 12
    OFFSET_RANGE_RATIO = 0.2
    DEFAULT_TEMPO = 600000

    SOURCES = {
        'original': [
            'Real Book'
        ],
        'improvised': [
            'Doug McKenzie',
            'Jazz-Midi',
            'Jazz Standards',
            'JazzPage',
            'MidKar',
            'Oocities',
            'Weimar DB'
        ]
    }

    def __init__(self, filepath, version):
        self.mido_obj = None
        self.music21_obj = None
        self.key = None
        self.mido_key = None
        self.music21_key = None
        self.music21_key2 = None
        self.tempo = None
        self.time_signature = None
        self.starting_measure = None
        self.song_structure = None
        self.chord_progression_key = None
        self.chord_progression_minor = None
        self.chord_progression_time_signature = None
        self.flat_chord_progression = []
        self.n_chord_prog_measures = None
        self.scale = None
        self.min_measure = None
        self.max_measure = None
        self.scale = None
        self.saved = False

        self.original_sources = ['Real Book']

        self.errors = []
        self.note_info = None
        self.split_note_info = []
        self.parts = []
        self.alignment_best_score = []
        self.alignment_all_scores = []
        self.candidate_score = None
        self.transpose_semitones = 0

        self.alignment_scores_folder = f'{src_path}/data/alignment_scores/v{version}'
        self.split_melody_data_folder = f'{src_path}/data/split_melody_data/v{version}'
        self.split_melody_folder = f'{src_path}/data/split_melody/v{version}'
        self.finalised_melody_folder = f'{src_path}/data/finalised'

        self.folder = filepath.split('/')[-3]
        self.source = filepath.split('/')[-2]
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        self.aligned_filepath = filepath.replace(self.folder, 'aligned_melodies')
        self.song_name = filepath_to_song_name(filepath)

        self.original = self.source in self.original_sources

    @staticmethod
    def remove_weak_polyphony(melody):
        new_melody = melody.copy()

        overlap = (new_melody['end_ticks'] - new_melody['ticks'].shift(-1)).clip(0, None)

        # skip last row as 'shift' messes it up
        new_melody.iloc[:-1, new_melody.columns.get_loc('duration')] -= overlap.iloc[:-1]
        new_melody.iloc[:-1, new_melody.columns.get_loc('end_ticks')] -= overlap.iloc[:-1]

        if is_weakly_polyphonic(new_melody):
            raise Exception('Error!!! Weak polyphony not removed correctly')

        return new_melody

    @staticmethod
    def remove_strong_polyphony(melody):
        new_melody = melody.copy()

        new_melody = new_melody\
            .sort_values('pitch', ascending=False)\
            .drop_duplicates('ticks')\
            .sort_values('ticks')

        if is_strongly_polyphonic(new_melody):
            raise Exception('Error!!! Strong polyphony not removed correctly')

        return new_melody

    @no_errors
    def setup(self):
        try:
            self.mido_obj = MidiFile(self.filepath)
            self.music21_obj = converter.parse(self.filepath)
            self.save_key()
            self.save_tempo()
            self.save_time_signature()
        except Exception as e:
            self.errors.append(f'error:{str(e)}')

    def set_song_structure(self, song_structure):
        self.song_structure = song_structure
        self.chord_progression_key = song_structure['key']
        self.chord_progression_minor = bool(song_structure['minor'])
        self.chord_progression_time_signature = tuple(song_structure['time_signature'])

        self.flat_chord_progression = flatten_chord_progression(self.song_structure)
        self.n_chord_prog_measures = int(len(self.flat_chord_progression) / self.chord_progression_time_signature[0])

        if self.chord_progression_time_signature != self.time_signature:
            self.errors.append("Chord progression time signature different from melody time signature ")

    @no_errors
    def save_key(self):
        self.music21_key = self.music21_obj.analyze('Krumhansl')
        self.music21_key2 = self.music21_obj.analyze('AardenEssen')

        for msg in self.mido_obj:
            if msg.type == 'key_signature':
                self.mido_key = music21.key.Key(msg.key)

        self.key = self.mido_key

        if self.key is None:
            self.key = self.music21_key

        if self.key is None:
            self.key = self.music21_key2

    @no_errors
    def save_tempo(self):
        tempos = []
        for i, track in enumerate(self.mido_obj.tracks):
            for msg in track:
                if msg.type == 'set_tempo':
                    tempo = msg.tempo
                    tempos.append(tempo)

        if len(tempos) == 0:
            self.errors.append('no tempo detected')
            self.tempo = self.DEFAULT_TEMPO
        elif len(tempos) == 1:
            self.tempo = tempos[0]
        else:
            if len(set(tempos)) <= 1:
                self.tempo = tempos[0]
            else:
                self.tempo = tempos[-1]  # TODO double-check if this is sensible
                # self.errors.append(f'multiple tempos: {tempos}')

    @no_errors
    def save_time_signature(self):
        time_signatures = []
        for i, track in enumerate(self.mido_obj.tracks):
            for msg in track:
                if msg.type == 'time_signature':
                    time_signature = (msg.numerator, msg.denominator)
                    time_signatures.append(time_signature)

        if len(time_signatures) == 0:
            self.errors.append(f'no time signature found')
        elif len(time_signatures) == 1:
            self.time_signature = time_signatures[0]
        else:
            if len(set(time_signatures)) <= 1:
                self.time_signature = time_signatures[0]
            else:
                self.errors.append(f'multiple time signatures {time_signatures}')

    def align_key(self):
        if self.key is None:
            self.errors.append('No key found!')
        else:
            inter = music21.interval.Interval(
                music21.pitch.Pitch(self.key.tonic),
                music21.pitch.Pitch(self.chord_progression_key)
            )

            if (self.key.mode == 'minor') == self.chord_progression_minor:
                self.transpose_semitones = inter.semitones
            else:
                if self.chord_progression_minor:
                    self.transpose_semitones = (inter.semitones + 3)
                else:
                    self.transpose_semitones = (inter.semitones - 3)

            self.transpose_semitones = ((self.transpose_semitones + 6) % 12) - 6

    @no_errors
    def parse_notes(self):
        total_time = 0
        total_ticks = 0
        tpb = self.mido_obj.ticks_per_beat

        notes_on = {}

        tpm = tpb * self.time_signature[0]
        ftpm = self.FINAL_TICKS_PER_BEAT * self.time_signature[0]

        note_info = []

        for m in self.mido_obj:
            ticks = int(np.round(mido.second2tick(m.time, tpb, self.tempo)))
            total_ticks += ticks
            total_time += m.time

            if m.type != 'note_on':
                continue

            if m.velocity > 0:
                if m.note not in notes_on:
                    notes_on[m.note] = []

                notes_on[m.note].append(total_ticks)
            else:
                if m.note in notes_on and len(notes_on[m.note]) > 0:
                    raw_note_ticks = notes_on[m.note][0] / tpm
                    quant_note_ticks = int(round(ftpm * raw_note_ticks))

                    raw_note_duration = (total_ticks - notes_on[m.note][0]) / tpm
                    quant_note_duration = int(round(ftpm * raw_note_duration))

                    note_offset = quant_note_ticks % ftpm

                    note_measure = int(np.floor(quant_note_ticks / ftpm))

                    del notes_on[m.note][0]

                    if quant_note_duration > 0:
                        pitch = m.note + self.transpose_semitones

                        note_info.append({
                            'pitch': pitch,
                            'pitch_class': replace_enharmonic(notes.int_to_note(pitch % 12)),
                            'time': total_time,
                            'raw_ticks': raw_note_ticks,
                            'quant_ticks': quant_note_ticks,
                            'raw_duration': raw_note_duration,
                            'quant_duration': quant_note_duration,
                            'offset': note_offset,
                            'measure': note_measure
                        })

        self.note_info = pd.DataFrame.from_dict(note_info)

        self.min_measure = self.note_info.measure.min()
        self.max_measure = int(np.ceil((self.note_info.measure + (
                (self.note_info.offset + self.note_info.quant_duration) / self.FINAL_TICKS_PER_BEAT /
                self.time_signature[0])).max())) - 1

    def transpose(self, transpose_semitones=None):
        if self.note_info is None:
            raise Exception('Note info not set yet')

        if transpose_semitones is not None:
            self.transpose_semitones = transpose_semitones

        self.note_info['pitch'] = self.note_info['pitch'] + self.transpose_semitones

    def calculate_alignment_score(self, offset):
        song_scores = []
        step = 1 / self.time_signature[0]
        ftpm = self.FINAL_TICKS_PER_BEAT * self.time_signature[0]
        position = offset * step

        for section in self.song_structure['sections']:
            for chord in self.song_structure['progression'][section]:
                pitches = self.note_info[
                    (self.note_info['measure'] + (self.note_info['offset'] / ftpm) >= position) &
                    (self.note_info['measure'] + (self.note_info['offset'] / ftpm) < position + step)
                    ]['pitch'].values

                scores = [self.chord_note_score(chord, pitch) for pitch in pitches]

                score = np.mean(scores) if len(scores) > 0 else 0.5
                song_scores.append(score)

                position += step

        return song_scores

    @no_errors
    def find_starting_measure(self):
        all_scores = []
        all_scores_mean = []
        step = 1 / self.time_signature[0]
        offset_range = int(
            (np.ceil(self.note_info.loc[self.note_info.shape[0] - 1, 'measure']) / step) * self.OFFSET_RANGE_RATIO)
        ftpm = self.FINAL_TICKS_PER_BEAT * self.time_signature[0]

        for offset in range(offset_range):
            position = offset * step
            song_scores = []

            for section in self.song_structure['sections']:
                for chord in self.song_structure['progression'][section]:
                    pitches = self.note_info[
                        (self.note_info['measure'] + (self.note_info['offset'] / ftpm) >= position) &
                        (self.note_info['measure'] + (self.note_info['offset'] / ftpm) < position + step)
                        ]['pitch'].values

                    scores = [self.chord_note_score(chord, pitch) for pitch in pitches]

                    score = np.mean(scores) if len(scores) > 0 else 0.33
                    song_scores.append(score)

                    position += step

            all_scores.append(song_scores)
            all_scores_mean.append(np.nanmean(song_scores))

        if not os.path.exists(os.path.join(self.alignment_scores_folder, self.source)):
            os.makedirs(os.path.join(self.alignment_scores_folder, self.source))

        plt.plot(all_scores_mean)
        plt.savefig(os.path.join(self.alignment_scores_folder, self.source, self.filename.replace('.mid', '.png')))
        # plt.close()
        # plt.cla()
        # plt.clf()

        candidates = np.array(sorted(range(len(all_scores_mean)), key=lambda i: all_scores_mean[i], reverse=True)[:12])

        candidate_measures = {}

        for candidate in candidates:
            candidate_measure = int(np.ceil(candidate * step))

            if candidate_measure not in candidate_measures:
                candidate_measures[candidate_measure] = 0

            candidate_measures[candidate_measure] += 1

        self.starting_measure = list(sorted(candidate_measures.items(), key=lambda item: item[1], reverse=True))[0][0]

        best_score_index = int(self.starting_measure / step)

        if best_score_index >= len(all_scores):
            best_score_index = 0

        self.alignment_best_score = all_scores[best_score_index]
        self.alignment_all_scores = all_scores_mean

    @no_errors
    def save_aligned_melody(self):
        first = True
        aligned_mido_obj = MidiFile()
        aligned_mido_obj.ticks_per_beat = self.mido_obj.ticks_per_beat

        for i, track in enumerate(self.mido_obj.tracks):
            starting_time = 0
            new_track = MidiTrack()
            aligned_mido_obj.tracks.append(new_track)
            for msg in track:
                starting_time += msg.time
                if msg.type == 'note_on':
                    if first:
                        msg.time = starting_time - \
                                   self.starting_measure * aligned_mido_obj.ticks_per_beat * self.time_signature[0]
                        if msg.time >= 0:
                            first = False
                else:
                    msg.time = 0

                if msg.time >= 0:
                    new_track.append(msg)

        try:
            aligned_mido_obj.save(self.aligned_filepath)
            self.saved = True
        except Exception as e:
            self.errors.append(e)

    @staticmethod
    def split_chord(chord_name):
        chord_obj = Chord(chord_name)
        chord_notes = [notes.int_to_note(x) for x in np.array(Chord(chord_name).getMIDI()) % 12]

        return chord_obj.root, chord_obj.bassnote, chord_notes

    @staticmethod
    def chord_score(c1, c2):
        if type(c1) == float or type(c2) == float:
            return np.nan

        notes_a = [notes.int_to_note(notes.note_to_int(x)) for x in chords.from_shorthand(c1)]
        notes_b = [notes.int_to_note(notes.note_to_int(x)) for x in chords.from_shorthand(c2)]
        root_a = notes_a[0]
        notes_set_a = set(notes_a)
        notes_set_b = set(notes_b)
        shared_a = notes_set_a.intersection(notes_set_b)
        shared_b = notes_set_b.intersection(notes_set_a)

        try:
            root_score = (len(notes_b) - notes_b.index(root_a)) / len(notes_b)
        except ValueError:
            root_score = 0

        notes_score_a = len(shared_a) / float(len(notes_a))
        notes_score_b = len(shared_b) / float(len(notes_b))
        notes_score = (1 / 2) * notes_score_a + (1 / 2) * notes_score_b
        score = ((1 / 5) * root_score) + ((4 / 5) * notes_score)

        return score

    def chord_note_score(self, chord, note):
        if type(chord) == float or type(note) == float:
            return np.nan

        chord_ints = Chord(chord).getMIDI()

        note = notes.int_to_note(note % 12)

        chord_note = [notes.int_to_note(chord_int % 12) for chord_int in chord_ints]

        if note in chord_note:
            return 1
        # elif note in self.scale:
        #     return 0.5
        else:
            return 0

    def save_split_melody(self, repetition, quantized, chords=True):
        split_melody_df = self.split_note_info[repetition-1]

        split_melody_data_folder = f'{self.split_melody_data_folder}/{self.source}'

        if not os.path.exists(split_melody_data_folder):
            os.makedirs(split_melody_data_folder)

        split_melody_data_filepath = f'{split_melody_data_folder}/{self.filename.replace(".mid", "")} ' \
                                     f'{"-o-" if self.original else "-" + str(repetition) + "-"}.csv'

        split_melody_df.to_csv(split_melody_data_filepath)

        split_melody_folder = f'{self.split_melody_folder}/{self.source}'

        if not os.path.exists(split_melody_folder):
            os.makedirs(split_melody_folder)

        split_melody_filepath = os.path.join(
            split_melody_folder,
            f'{self.filename.replace(".mid", "")} {"-o-" if self.original else f"-{repetition}-"}.mid'
        )
        if chords:
            notes_and_chord_to_midi(split_melody_df, self.song_structure, quantized, split_melody_filepath)
        else:
            notes_to_midi(split_melody_df, split_melody_filepath)

    def split_melody(self, quantized, save=True):
        bpm = self.time_signature[0]
        ftpm = self.FINAL_TICKS_PER_BEAT * bpm
        starting_measure = self.starting_measure
        n_measures = self.note_info['measure'].max() + 1

        repetition = 1
        lower_bound = starting_measure
        upper_bound = starting_measure + self.n_chord_prog_measures

        while lower_bound <= n_measures:
            valid_notes = self.note_info[
                (self.note_info['measure'] >= lower_bound) &
                (self.note_info['measure'] < upper_bound)
            ]

            # print(f"Repetition {repetition}, {len(valid_notes)} notes")

            if self.original or len(valid_notes) >= 40:
                notes_df = pd.DataFrame(valid_notes)

                notes_df['measure'] = (
                        notes_df['measure'] -
                        starting_measure -
                        (self.n_chord_prog_measures * (repetition - 1))
                )

                notes_df['raw_ticks'] = (
                        notes_df['raw_ticks'] -
                        starting_measure -
                        self.n_chord_prog_measures * (repetition - 1)
                )

                notes_df['quant_ticks'] = (
                        notes_df['quant_ticks'] -
                        ftpm * starting_measure -
                        ftpm * self.n_chord_prog_measures * (repetition - 1)
                )

                if quantized:
                    notes_df['ticks'] = notes_df['quant_ticks']
                    notes_df['duration'] = notes_df['quant_duration']
                else:
                    notes_df['ticks'] = notes_df['raw_duration']
                    notes_df['duration'] = notes_df['quant_duration']

                flat_chord_progression = self.flat_chord_progression

                def get_current_chord(row):
                    offset = row['measure'] + (row['offset'] / ftpm)
                    chord_idx = int(np.floor(offset * bpm))

                    return flat_chord_progression[chord_idx]

                current_chords = notes_df.apply(lambda x: get_current_chord(x), axis=1)
                notes_df['chord_name'] = current_chords

                self.split_note_info.append(notes_df)

                if save:
                    self.save_split_melody(repetition, quantized)

            repetition += 1
            lower_bound += self.n_chord_prog_measures
            upper_bound += self.n_chord_prog_measures

            if self.original:
                return True

        return True

    def chord_progression_comparison(self):
        n_measures = int(self.note_info['measure'].max() + 1)

        outro_init = (self.max_measure - self.min_measure + 1) % self.n_chord_prog_measures
        outro_auto = (self.max_measure - self.starting_measure + 1) % self.n_chord_prog_measures

        return {
            'cp_measures': self.n_chord_prog_measures,
            'n_measures': n_measures,
            'repetitions': n_measures // self.n_chord_prog_measures,
            'min_measure': self.min_measure,
            'max_measure': self.max_measure,
            'outro_init': outro_init,
            'outro_auto': outro_auto,
            'outro_difference': outro_init - outro_auto
        }

    @no_errors
    def align_melody(self):
        self.align_key()
        self.parse_notes()
        self.find_starting_measure()
        self.save_aligned_melody()

    @no_errors
    def manually_align(self, starting_measure, quantized=True):
        self.starting_measure = starting_measure

        self.align_key()
        self.parse_notes()
        self.split_melody(quantized)
