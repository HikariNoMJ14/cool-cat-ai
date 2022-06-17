import os

dir_path = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.join(dir_path, '..', '..')

REST_PITCH_SYMBOL = 128
REST_ATTACK_SYMBOL = 2
OCTAVE_SEMITONES = 12
TICKS_PER_MEASURE = 48
PITCH_CLS = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
REST_VAL = -1

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

INPUT_DATA_FOLDER = os.path.join(
    src_path,
    'data',
    'finalised',
    'csv',
)
