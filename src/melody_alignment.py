import os
import json
import re
from glob import glob

import music21
from music21 import converter

import mido
from mido import MidiFile, MidiTrack

from mingus.core import chords
import mingus.core.notes as notes

from omnizart.chord import app as capp
from omnizart.utils import synth_midi

import numpy as np
from matplotlib import pyplot as plt


def find_chords(filename):
    synth_filename = filename.replace('standardized', 'synthesized').replace('mid', 'wav')

    synth_midi(filename, synth_filename)

    output_filename = filename.replace('standardized', 'chords')

    capp.transcribe(synth_filename, output=output_filename)

    return output_filename


def no_errors(func):
    def inner(*args):
        if len(args[0].errors) == 0:
            func(*args)
        else:
            print(args[0].errors)
    return inner


class Melody:
    mido_obj = None
    music21_obj = None
    key = None
    tempo = None
    time_signature = None
    starting_measure = None
    song_structure = None
    chord_progression_key = None
    chord_progression_minor = None

    alignment_scores_folder = '../data/alignment_scores'

    def __init__(self, filepath):
        self.errors = []
        self.note_info = []
        self.parts = []
        self.alignment_score = {}
        self.transpose_semitones = 0

        song_name = os.path.basename(filepath).replace('.mid', '')
        song_name = "".join(song_name.split(' - ')[-1])
        song_name = re.sub('\(.*\)', '', song_name).strip()

        self.folder = filepath.split('/')[-3]
        self.source = filepath.split('/')[-2]
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        self.aligned_filepath = filepath.replace(self.folder, 'aligned_melodies')
        self.song_name = song_name

    @no_errors
    def setup(self):
        try:
            self.mido_obj = MidiFile(self.filepath)
            self.music21_obj = converter.parse(self.filepath)
            self.save_key()
            self.save_tempo()
            self.save_time_signature()
        except Exception as e:
            self.errors.append(e)

    @no_errors
    def set_song_structure(self, song_structure):
        self.song_structure = song_structure
        self.chord_progression_key = song_structure['key']
        self.chord_progression_minor = bool(song_structure['minor'])

    def save_key(self):
        self.key = self.music21_obj.analyze('key')

    def save_tempo(self):
        tempos = []
        for i, track in enumerate(self.mido_obj.tracks):
            for msg in track:
                if msg.type == 'set_tempo':
                    tempo = msg.tempo
                    tempos.append(tempo)

        if len(tempos) == 0:
            self.errors.append('no tempo detected')
            self.tempo = 500000
        elif len(tempos) == 1:
            self.tempo = tempos[0]
        else:
            if len(set(tempos)) <= 1:
                self.tempo = tempos[0]
            else:
                self.errors.append(f'multiple tempos: {tempos}')

    def save_time_signature(self):
        time_signatures = []
        for i, track in enumerate(self.mido_obj.tracks):
            for msg in track:
                if msg.type == 'time_signature':
                    time_signature = (msg.numerator, msg.denominator)
                    time_signatures.append(time_signature)

        if len(time_signatures) == 0:
            self.time_signature = (4 / 4)
        elif len(time_signatures) == 1:
            self.time_signature = time_signatures[0]
        else:
            if len(set(time_signatures)) <= 1:
                self.time_signature = time_signatures[0]
            else:
                self.errors.append(f'multiple time signatures {time_signatures}')

    @no_errors
    def align_key(self):
        inter = music21.interval.Interval(
            music21.pitch.Pitch(self.key.tonic),
            music21.pitch.Pitch(self.chord_progression_key)
        )

        if (self.key.mode == 'minor') == self.chord_progression_minor:
            self.transpose_semitones = inter.semitones % 12

        else:
            self.errors.append("mode doesn't match")
            # print("mode doesn't match")

            if self.chord_progression_minor:
                self.transpose_semitones = (inter.semitones + 3) % 12
            else:
                self.transpose_semitones = (inter.semitones - 3) % 12

    @no_errors
    def parse_notes(self):
        time = 0
        tpb = self.mido_obj.ticks_per_beat

        notes_on = {}

        for m in self.mido_obj:
            ticks = int(np.round(mido.second2tick(m.time, tpb, self.tempo)))
            time += ticks

            if m.type == 'note_on':
                if m.velocity > 0:
                    if m.note not in notes_on:
                        notes_on[m.note] = []

                    notes_on[m.note].append(time)
                else:
                    if m.note in notes_on and len(notes_on[m.note]) > 0:
                        note = {
                            'pitch': m.note + self.transpose_semitones,
                            'ticks': notes_on[m.note][0],
                            'offset': notes_on[m.note][0] % (tpb * self.time_signature[1]),
                            'measure': notes_on[m.note][0] / (tpb * self.time_signature[1]),
                            'duration': time - notes_on[m.note][0]
                        }
                        del notes_on[m.note][0]
                        self.note_info.append(note)

    @no_errors
    def find_starting_measure(self):
        all_scores = []
        step = 1 / self.time_signature[1]
        offset_range = int(np.ceil(self.note_info[-1]['measure']) / (step * 2))

        for offset in range(-offset_range, offset_range):
            measure = offset * step
            song_scores = []

            for section in self.song_structure['sections']:
                for chord in self.song_structure['progression'][section]:
                    pitches = [x['pitch'] for x in self.note_info
                               if measure <= x['measure'] < measure + step]

                    scores = []
                    for pitch in pitches:
                        scores.append(self.chord_note_score(chord, pitch))

                    score = np.mean(scores) if len(scores) > 0 else 0.5
                    song_scores.append(score)

                    measure += step

            song_score = np.nanmean(song_scores)
            all_scores.append(song_score)

        plt.plot(range(-offset_range, offset_range), all_scores)
        plt.savefig(os.path.join(self.alignment_scores_folder, self.source, self.filename.replace('.mid', '.png')))
        plt.close()
        plt.cla()
        plt.clf()

        found = False
        candidate_rank = 1
        while not found:
            top_idx = np.array(sorted(range(len(all_scores)),
                                      key=lambda i: all_scores[i])[-candidate_rank]) - offset_range
            if top_idx % step == 0:
                found = True
                self.starting_measure = int(np.ceil(top_idx * step))
                self.alignment_score['candidate_rank'] = int(candidate_rank)
                self.alignment_score['candidate'] = int(top_idx)
                self.alignment_score['mean_score'] = float(np.mean(all_scores))
                self.alignment_score['max_score'] = float(np.max(all_scores))

                q_score_1 = 1 - 1 / (len(all_scores) * 10)
                q_score_2 = 1 - 1 / (len(all_scores) * 2)
                q_score_3 = 1 - 1 / (len(all_scores))
                q_score_4 = 1 - 1 / (len(all_scores) / 2)

                self.alignment_score['outlier_score_1'] = float(
                    np.quantile(all_scores, q_score_1) -
                    np.quantile(all_scores, q_score_2))
                self.alignment_score['outlier_score_2'] = float(
                    np.quantile(all_scores, q_score_1) -
                    np.quantile(all_scores, q_score_3))
                self.alignment_score['outlier_score_3'] = float(
                    np.quantile(all_scores, q_score_1) -
                    np.quantile(all_scores, q_score_4))
            else:
                candidate_rank += 1
                if candidate_rank % 100 == 0:
                    print(candidate_rank)

    @no_errors
    def save_aligned_melody(self):
        first = True
        aligned_mido_obj = MidiFile()
        aligned_mido_obj.ticks_per_beat = self.mido_obj.ticks_per_beat

        starting_time = 0
        for i, track in enumerate(self.mido_obj.tracks):
            new_track = MidiTrack()
            aligned_mido_obj.tracks.append(new_track)
            for msg in track:
                starting_time += msg.time
                if msg.type == 'note_on':
                    if first:
                        msg.time = starting_time - \
                                   self.starting_measure * aligned_mido_obj.ticks_per_beat * self.time_signature[1]
                        first = False
                else:
                    msg.time = 0
                new_track.append(msg)

        aligned_mido_obj.save(self.aligned_filepath)

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

        # print(c1, c2)
        # print(notes_a, notes_b)
        # print(root_a, root_b)
        # print(notes_set_a, notes_set_b)
        # print(shared_a, shared_b)
        # print(root_score, notes_score)
        # print(score)
        # print('-------')

        return score

    @staticmethod
    def chord_note_score(c, n):
        if type(c) == float or type(n) == float:
            return np.nan

        found = False
        while not found:
            try:
                c = chords.from_shorthand(c)
                found = True
            except Exception as e:
                if len(c) > 0:
                    c = c[:-1]
                else:
                    raise Exception("Chord can't be found")

        n = notes.int_to_note(n % 12)

        notes_c = [notes.int_to_note(notes.note_to_int(x)) for x in c]

        # if key.mode == 'major':
        #     self.key_scale = music21.scale.MajorScale(key.tonic)
        # elif key.mode == 'minor':
        #     self.key_scale = music21.scale.MinorScale(key.tonic)

        if n in notes_c:
            return 1
        else:
            return 0

    # @no_errors
    def align_melody(self):
        self.align_key()
        self.parse_notes()
        self.find_starting_measure()
        # self.save_aligned_melody()


if __name__ == "__main__":
    from datetime import datetime

    folder = '/media/manu/DATA/Mac/Documents/University/Thesis/Complete Examples Melodies'

    filepaths = [y for x in os.walk(folder) for y in glob(os.path.join(x[0], '*.mid'))]

    # filepaths = [
    #     # '../data/Complete Examples Melodies/Jazz-Midi/All Of Me.mid',
    #     '../data/Complete Examples Melodies/Real Book/All Blues.mid'
    # ]

    chord_progressions_filepath = '../data/chord_progressions/irb_chord_progressions.json'
    chord_progressions = json.load(open(chord_progressions_filepath))

    song_scores = {}

    for fp in filepaths:
        melody = Melody(fp)

        if melody.song_name in chord_progressions:
            melody.setup()
            melody.set_song_structure(chord_progressions[melody.song_name])
            melody.align_melody()

            key = os.path.join(melody.source, melody.filename)
            song_scores[key] = melody.alignment_score.copy()

            del melody

    json.dump(song_scores,
              open(f'../data/alignment_scores/song_scores-{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}.json', 'w+'))
