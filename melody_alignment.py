import os.path
import numpy as np

import mido
from mido import MidiFile
from mingus.core import chords
import mingus.core.notes as notes
from omnizart.chord import app as capp
from omnizart.utils import synth_midi


class MelodyAlignment:

    def __init__(self, filename):
        pass




class Song:

    tempo = None
    time_signature = None
    note_info = []

    parts = []

    def __init__(self, filename):
        self.mido_obj = MidiFile(filename)
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
                    #             print(notes_on)
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


def chord_score(c1, c2):
    if type(c1) == float or type(c2) == float:
        return np.nan

    notes_a = [notes.int_to_note(notes.note_to_int(x)) for x in chords.from_shorthand(c1)]
    notes_b = [notes.int_to_note(notes.note_to_int(x)) for x in chords.from_shorthand(c2)]
    root_a = notes_a[0]
    root_b = notes_b[0]
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


def find_chords(filename):
    synth_filename = filename.replace('standardized', 'synthesized').replace('mid', 'wav')

    synth_midi(filename, synth_filename)

    output_filename = filename.replace('standardized', 'chords')

    capp.transcribe(synth_filename, output=output_filename)

    return output_filename




