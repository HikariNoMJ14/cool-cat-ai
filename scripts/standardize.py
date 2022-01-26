import music21
import numpy as np
from mido import MidiFile, MidiTrack


def standardize(filename):
    transposed_filename = filename.replace('Complete Examples', 'transposed')
    # standardized_filename = filename.replace('Complete Examples', 'standardized')

    transpose_to_c(filename, transposed_filename)
    # change_tempo_120(transposed_filename, standardized_filename)

    return transposed_filename


def transpose_to_c(src_filename, dst_filename):

    old_stream = music21.converter.parse(src_filename)
    old_key = old_stream.analyze('key')

    if old_key.mode == 'major':
        base = 'C'
    else:
        base = 'A'

    inter = music21.interval.Interval(music21.pitch.Pitch(base), old_key.tonic)
    old_stream.transpose(-inter.semitones, inPlace=True)

    print(old_key, inter.semitones)

    old_stream.write('midi', fp=dst_filename)


def change_tempo_120(src_filename, dst_filename):
    mid = MidiFile(src_filename)
    new_mid = MidiFile()
    new_mid.ticks_per_beat = mid.ticks_per_beat
    print(len(mid.tracks))
    for i, track in enumerate(mid.tracks):
        new_track = MidiTrack()
        new_mid.tracks.append(new_track)
        for msg in track:
            if msg.type == 'set_tempo':
                print(i, msg)
                msg.tempo = 500000
                # print(msg)

            new_track.append(msg)

    new_mid.save(dst_filename)
