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

from ezchord import Chord

from omnizart.chord import app as capp
from omnizart.utils import synth_midi

import numpy as np
import pandas as pd
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
            print(args[0].song_name, args[0].errors)

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
    chord_progression_time_signature = None
    scale = None
    saved = False

    final_ticks_per_beats = 12

    original_sources = ['Real Book']
    alignment_scores_folder = '../data/alignment_scores'

    def __init__(self, filepath):
        self.errors = []
        self.note_info = []
        self.parts = []
        self.alignment_best_score = []
        self.alignment_all_scores = []
        self.candidate_score = None
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

        self.original = self.source in self.original_sources

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

        if self.chord_progression_time_signature != self.time_signature:
            self.errors.append("Chord progression time signature different from melody time signature ")

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
            self.errors.append(f'no time signature found')
        elif len(time_signatures) == 1:
            self.time_signature = time_signatures[0]
        else:
            if len(set(time_signatures)) <= 1:
                self.time_signature = time_signatures[0]
            else:
                self.errors.append(f'multiple time signatures {time_signatures}')

    def align_key(self):
        inter = music21.interval.Interval(
            music21.pitch.Pitch(self.key.tonic),
            music21.pitch.Pitch(self.chord_progression_key)
        )

        if (self.key.mode == 'minor') == self.chord_progression_minor:
            self.transpose_semitones = inter.semitones % 12

        else:
            if self.chord_progression_minor:
                self.transpose_semitones = (inter.semitones + 3) % 12
            else:
                self.transpose_semitones = (inter.semitones - 3) % 12

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
                            'offset': notes_on[m.note][0] % (tpb * self.time_signature[0]),
                            'measure': notes_on[m.note][0] / (tpb * self.time_signature[0]),
                            'duration': time - notes_on[m.note][0]
                        }
                        del notes_on[m.note][0]
                        self.note_info.append(note)

    def find_starting_measure(self):
        all_scores = []
        all_scores_mean = []
        step = 1 / self.time_signature[0]
        offset_range = int(np.ceil(self.note_info[-1]['measure']) / step)

        if not self.chord_progression_minor:
            self.scale = music21.scale.MajorScale(self.chord_progression_key)
        else:
            self.scale = music21.scale.MinorScale(self.chord_progression_key)

        self.scale = set([x.name for x in self.scale.getPitches()])

        for offset in range(offset_range):
            measure = offset * step
            song_scores = []

            for section in self.song_structure['sections']:
                for chord in self.song_structure['progression'][section]:
                    pitches = [x['pitch'] for x in self.note_info
                               if measure <= x['measure'] < measure + step]

                    scores = [self.chord_note_score(chord, pitch) for pitch in pitches]

                    score = np.mean(scores) if len(scores) > 0 else 0.33
                    song_scores.append(score)

                    measure += step

            all_scores.append(song_scores)
            all_scores_mean.append(np.nanmean(song_scores))

        plt.plot(all_scores_mean)
        plt.savefig(os.path.join(self.alignment_scores_folder, self.source, self.filename.replace('.mid', '.png')))
        plt.close()
        plt.cla()
        plt.clf()

        candidates = np.array(sorted(range(len(all_scores_mean)), key=lambda i: all_scores_mean[i], reverse=True)[:12])

        candidate_measures = {}

        for candidate in candidates:
            candidate_measure = int(np.ceil(candidate * step))

            if candidate_measure not in candidate_measures:
                candidate_measures[candidate_measure] = 0

            candidate_measures[candidate_measure] += 1

        self.starting_measure = list(sorted(candidate_measures.items(), key=lambda item: item[1], reverse=True))[0][0]
        self.alignment_best_score = all_scores[int(self.starting_measure / step)]
        self.alignment_all_scores = all_scores_mean

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

    @no_errors
    def align_melody(self):
        self.align_key()
        self.parse_notes()
        self.find_starting_measure()

        # if np.max(self.alignment_score) >= 0.666 and self.starting_measure < 16:
        self.save_aligned_melody()

    @no_errors
    def split_melody(self):
        bpm = self.time_signature[0]
        tpb = self.mido_obj.ticks_per_beat
        ftpb = self.final_ticks_per_beats
        starting_measure = self.starting_measure

        def normalize(col):
            return ((col / tpb) * ftpb * bpm).apply(round).apply(int)

        n_chord_prog_measures = int(np.sum([
            len(self.song_structure['progression'][section])
            for section in self.song_structure['sections']]
        ) / bpm)

        n_measures = int(np.floor(np.max([
            x['measure'] + ((x['duration'] / tpb) / bpm) for x in self.note_info
        ])) + 1)

        repetition = 1
        lower_bound = starting_measure
        upper_bound = starting_measure + n_chord_prog_measures

        while upper_bound <= n_measures:
            valid_notes = [n for n in self.note_info if
                           lower_bound <= n['measure'] < upper_bound]

            notes_df = pd.DataFrame(valid_notes)

            notes_df['ticks'] -= (notes_df['ticks'].min() - notes_df['ticks'].min() % tpb)

            notes_df['ticks'] = normalize(notes_df['ticks'])
            notes_df['offset'] = normalize(notes_df['offset'])
            notes_df['duration'] = normalize(notes_df['duration'])
            notes_df['measure'] = notes_df['measure'].apply(round).apply(int)

            notes_df.to_csv(f'../data/split_melody/{self.source}/'
                            f'{self.filename.replace(".mid", "")}_'
                            f'{"original" if self.original else repetition}.csv')

            repetition += 1
            lower_bound += n_chord_prog_measures
            upper_bound += n_chord_prog_measures

            if self.original:
                return True

        return True


if __name__ == "__main__":
    from datetime import datetime

    folder = '../data/Complete Examples Melodies/Real Book'

    # filepaths = [y for x in os.walk(folder) for y in glob(os.path.join(x[0], '*.mid'))]

    filepaths = [
        # '../data/Complete Examples Melodies/Jazz-Midi/All Of Me.mid',
        # '../data/Complete Examples Melodies/Real Book/All Of Me.mid',
        # '../data/Complete Examples Melodies/Real Book/All Blues.mid',
        # '../data/Complete Examples Melodies/Oocities/Come Rain Or Come Shine.mid'
        # '../data/Complete Examples Melodies/Real Book/A Felicidade.mid'
        # '../data/Complete Examples Melodies/Real Book/Afro Blue.mid'
        # '../data/Complete Examples Melodies/Real Book/Autumn Leaves.mid'
        # '../data/Complete Examples Melodies/Real Book/Ornithology.mid'
        # '../data/Complete Examples Melodies/Real Book/Margie.mid',
        # '../data/Complete Examples Melodies/Real Book/Oleo.mid',
        # '../data/Complete Examples Melodies/Real Book/Blue Train.mid',
        # '../data/Complete Examples Melodies/Weimar DB/Bix Beiderbecke - Margie.mid',
        # '../data/Complete Examples Melodies/Weimar DB/Red Garland - Oleo.mid',
        # '../data/Complete Examples Melodies/Weimar DB/Miles Davis - Oleo (1).mid',
        # '../data/Complete Examples Melodies/Weimar DB/Miles Davis - Oleo (2).mid',
        # '../data/Complete Examples Melodies/Weimar DB/John Coltrane - Oleo.mid',
        # '../data/Complete Examples Melodies/Weimar DB/Lee Morgan - Blue Train.mid',
        # '../data/Complete Examples Melodies/Weimar DB/Curtis Fuller - Blue Train.mid',
        # '../data/Complete Examples Melodies/Weimar DB/John Coltrane - Blue Train.mid'
    ]
    chord_progressions = {}

    irb_chord_progressions_filepath = '../data/chord_progressions/irb_chord_progressions.json'
    wdb_chord_progressions_filepath = '../data/chord_progressions/weimar_db.json'
    manual_chord_progressions_filepath = '../data/chord_progressions/manual_chord_progressions.json'

    chord_progressions.update(json.load(open(irb_chord_progressions_filepath)))
    chord_progressions.update(json.load(open(wdb_chord_progressions_filepath)))
    chord_progressions.update(json.load(open(manual_chord_progressions_filepath)))

    song_scores = {}

    no_chords = []
    errors = {}

    for fp in filepaths:
        melody = Melody(fp)
        key = os.path.join(melody.source, melody.filename)

        if melody.song_name in chord_progressions:
            melody.setup()

            if melody.time_signature is not None and melody.time_signature[0] == 4:
                melody.set_song_structure(chord_progressions[melody.song_name])
                melody.align_melody()

                print(melody.song_name, melody.starting_measure)

                melody.split_melody()

                song_scores[key] = {
                    'melody_key': melody.key.name,
                    'chord_progression_key': melody.chord_progression_key,
                    'starting_measure': melody.starting_measure,
                    'best_score': melody.alignment_best_score,
                    'all_scores': melody.alignment_all_scores
                }
        else:
            no_chords.append(key)

        if len(melody.errors) > 0:
            errors[key] = melody.errors

        del melody

    json.dump(song_scores,
              open(f'../data/alignment_scores/song_scores-{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}.json', 'w+'))

    json.dump(no_chords,
              open(f'../data/alignment_scores/no_chords-{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}.json', 'w+'))

    json.dump(errors,
              open(f'../data/alignment_scores/errors-{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}.json', 'w+'))
