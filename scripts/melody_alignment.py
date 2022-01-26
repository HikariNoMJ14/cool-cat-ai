import os.path
import numpy as np
from matplotlib import pyplot as plt

import mido
from mido import MidiFile, MidiTrack
from jchord.progressions import ChordProgression, Song, SongSection
from mingus.core import chords
import mingus.core.notes as notes
from omnizart.chord import app as capp
from omnizart.utils import synth_midi


def find_chords(filename):
    synth_filename = filename.replace('standardized', 'synthesized').replace('mid', 'wav')

    synth_midi(filename, synth_filename)

    output_filename = filename.replace('standardized', 'chords')

    capp.transcribe(synth_filename, output=output_filename)

    return output_filename


class Melody:

    tempo = None
    time_signature = None
    note_info = []
    parts = []
    starting_measure = None

    def __init__(self, filepath):
        folder = filepath.split('/')[-3]
        self.aligned_filepath = filepath.replace(folder, 'aligned_melodies')

        self.mido_obj = MidiFile(filepath)
        self.save_tempo()
        self.save_time_signature()

    def save_tempo(self):
        tempos = []
        for i, track in enumerate(self.mido_obj.tracks):
            for msg in track:
                if msg.type == 'set_tempo':
                    tempo = msg.tempo
                    tempos.append(tempo)

        if len(tempos) == 0:
            print('no tempo detected')
            self.tempo = 500000
        elif len(tempos) == 1:
            self.tempo = tempos[0]
        else:
            print('multiple tempos', tempos)
            self.tempo = np.mean(tempos)

    def save_time_signature(self):
        time_signatures = []
        for i, track in enumerate(self.mido_obj.tracks):
            for msg in track:
                if msg.type == 'time_signature':
                    time_signature = (msg.numerator, msg.denominator)
                    time_signatures.append(time_signature)

        if len(time_signatures) == 0:
            self.time_signature = (4/4)
        elif len(time_signatures) == 1:
            self.time_signature = time_signatures[0]
        else:
            print('multiple time signatures', time_signatures)
            self.time_signature = time_signatures[0]

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
                    if m.note in notes_on:
                        note = {
                            'pitch': m.note,
                            'ticks': notes_on[m.note][0],
                            'offset': notes_on[m.note][0] % (tpb * self.time_signature[1]),
                            'measure': notes_on[m.note][0] / (tpb * self.time_signature[1]),
                            'duration': time - notes_on[m.note][0]
                        }
                        del notes_on[m.note][0]
                        self.note_info.append(note)

    def find_starting_measure(self):
        all_scores = []
        step = 1/self.time_signature[1]
        offset_range = int(np.ceil(self.note_info[-1]['measure']) / (step * 2))

        for offset in range(-offset_range, offset_range):
            measure = offset * step
            song_scores = []

            for section in song.sections:
                for chord in section.progression.progression:
                    pitches = [x['pitch'] for x in self.note_info
                               if measure <= x['measure'] < measure + step]

                    scores = []
                    for pitch in pitches:
                        scores.append(self.chord_note_score(chord.name, pitch))

                    score = np.mean(scores) if len(scores) > 0 else 0.5
                    song_scores.append(score)

                    measure += step

            song_score = np.nanmean(song_scores)
            all_scores.append(song_score)

        plt.plot(all_scores)
        # plt.show()

        found = False
        j = 1
        while not found:
            top_idx = np.array(sorted(range(len(all_scores)),
                                      key=lambda i: all_scores[i])[-j]) - offset_range
            if top_idx % step == 0:
                found = True
                self.starting_measure = int(np.ceil(top_idx * step))
            else:
                j += 1

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

        c = chords.from_shorthand(c)
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


def align_melody(filename):
    pass
    melody = Melody(filename)

    melody.parse_notes()

    melody.find_starting_measure()

    melody.save_aligned_melody()


if __name__ == "__main__":
    f = './data/Complete Examples Melodies/Jazz-Midi/All Of Me.mid'

    chord_prog = {
        'A': 'Cmaj7 -- -- -- -- -- -- -- E7 -- -- -- -- -- -- -- '
             'A7 -- -- -- -- -- -- -- Dm7 -- -- -- -- -- -- --',
        'B': 'E7 -- -- -- -- -- -- -- Am7 -- -- -- -- -- -- -- '
             'D7 -- -- -- -- -- -- -- Dm7 -- -- -- G7 -- -- --',
        'C': 'Fmaj7 -- -- -- Fm6 -- -- -- Em7 -- -- -- A7 -- -- -- '
             'Dm7 -- -- -- G7 -- -- -- C6 -- Ebdim7 -- Dm7 -- G7 --',
    }
    parts = ['A', 'B', 'A', 'C']

    sections = []

    for part in parts:
        progression = ChordProgression.from_string(chord_prog[part])
        ss = SongSection(name=part, progression=progression)
        sections.append(ss)

    song = Song(sections)

    align_melody(f)

