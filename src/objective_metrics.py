import os
from glob import glob
import random, itertools

import scipy.stats
import numpy as np
import pandas as pd

N_PITCH_CLS = 12  # {C, C#, ..., Bb, B}
TICKS_PER_MEASURE = 48


def compute_histogram_entropy(histogram):
    return scipy.stats.entropy(histogram) / np.log(2)


def get_pitch_histogram(melody, pitches=range(128), verbose=False):
    pitches = [x for x in melody['pitch'] if x in pitches]

    if not len(melody):
        if verbose:
            print('The sequence contains no notes.')
        return None

    pitches = pd.Series(pitches) % N_PITCH_CLS

    histogram = pitches.value_counts(normalize=True)

    hist = np.zeros((N_PITCH_CLS,))
    for i in range(N_PITCH_CLS):
        if i in histogram.index:
            hist[i] = histogram.loc[i]

    return hist


def get_onset_xor_distance(sequence_a, sequence_b):
    def make_onset_vector(seq):
        onset_vector = np.zeros((TICKS_PER_MEASURE,))

        for ev in seq:
            onset_vector[ev] = 1

        return onset_vector

    a_onsets, b_onsets = make_onset_vector(sequence_a), make_onset_vector(sequence_b)

    dist = np.sum(np.abs(a_onsets - b_onsets)) / TICKS_PER_MEASURE

    # print(a_onsets, b_onsets, np.sum(np.abs(a_onsets - b_onsets)))

    return dist


def get_bars_crop(melody, start_measure, end_measure):
    if start_measure < 0 or end_measure < 0:
        raise ValueError('Invalid start_measure: {}, or end_measure: {}.'.format(start_measure, end_measure))

    return melody[(melody['measure'] >= start_measure) & (melody['measure'] <= end_measure)]


def compute_piece_pitch_entropy(melody, window_size, pitches=range(128), verbose=False):
    n_bars = melody['measure'].max() + 1

    pitch_entropies = []
    for start_measure in range(1, n_bars - window_size + 2):
        sequence = get_bars_crop(melody, start_measure, start_measure + window_size - 1)
        pitch_histogram = get_pitch_histogram(sequence, pitches=pitches)

        if pitch_histogram is None:
            if verbose:
                print('No notes in this crop: {}~{} measures.'.format(start_measure, start_measure + window_size - 1))
            continue

        pitch_entropies.append(compute_histogram_entropy(pitch_histogram))

    return np.mean(pitch_entropies)


def compute_piece_groove_similarity(melody, max_pairs=1000):
    n_bars = melody['measure'].max() + 1

    bar_sequences = []
    for b in range(1, n_bars + 1):
        bar_sequences.append(get_bars_crop(melody, b, b))

    pairs = list(itertools.combinations(range(n_bars), 2))
    if len(pairs) > max_pairs:
        pairs = random.sample(pairs, max_pairs)

    groove_similarities = []
    for p in pairs:
        groove_similarities.append(
            1. - get_onset_xor_distance(bar_sequences[p[0]]['offset'], bar_sequences[p[1]]['offset'])
        )

    return np.mean(groove_similarities)


def compute_piece_chord_progression_irregularity(chord_sequence, ngram=3):
    if len(chord_sequence) <= ngram:
        return 1.

    num_ngrams = len(chord_sequence) - ngram
    unique_set = set()

    for i in range(num_ngrams):
        str_repr = '_'.join(['-'.join(str(x)) for x in chord_sequence[i: i + ngram]])

        if str_repr not in unique_set:
            unique_set.add(str_repr)

    return len(unique_set) / num_ngrams


# def compute_structure_indicator(mat_file, low_bound_sec=0, upp_bound_sec=128, sample_rate=2):
#     '''
#     Computes the structureness indicator SI(low_bound_sec, upp_bound_sec) from fitness scape plot (stored in a MATLAB .mat file).
#     (Metric ``SI``)
#
#     Parameters:
#       mat_file (str): path to the .mat file containing fitness scape plot of a piece. (computed by ``run_matlab_scapeplot.py``).
#       low_bound_sec (int, >0): the smallest timescale (in seconds) you are interested to examine.
#       upp_bound_sec (int, >0): the largest timescale (in seconds) you are interested to examine.
#       sample_rate (int): sample rate (in Hz) of the input fitness scape plot.
#
#     Returns:
#       float: 0~1, the structureness indicator (i.e., max fitness value) of the piece within the given range of timescales.
#     '''
#
#     assert low_bound_sec > 0 and upp_bound_sec > 0, '`low_bound_sec` and `upp_bound_sec` should be positive, got: low_bound_sec={}, upp_bound_sec={}.'.format(
#         low_bound_sec, upp_bound_sec)
#     low_bound_ts = int(low_bound_sec * sample_rate) - 1
#     upp_bound_ts = int(upp_bound_sec * sample_rate)
#     f_mat = read_fitness_mat(mat_file)
#
#     if low_bound_ts >= f_mat.shape[0]:
#         score = 0
#     else:
#         score = np.max(f_mat[low_bound_ts: upp_bound_ts])
#
#     return score


# Mode Collapse Metrics

# We use the following metrics to evaluate the general
# quality of musical generations. The QR and T S metrics,
# as described below, have been adapted from (Dong et al.
# 2018). We propose additional metrics (CP R, DP R, OR)
# to address concerns of repeated notes and observational bias
# in generated rhythms. To see how well the generator model
# performs, we compare these metrics on a set of generated
# sequences against the training corpus.

# Qualified Rhythm frequency (QR)
# QR measures the frequency of note durations within valid beat ratios of
# {1, 1/2, 1/4, 1/8, 1/16}, their dotted and triplet counter-
# parts, and any tied combination of two valid ratios. This
# generalizes beyond MuseGAN’s qualified note metric,
# which only measures the frequency of durations greater
# than a 32nd note.

# Consecutive Pitch Repetitions (CPR)
# For a specified length l, CPR measures the frequency of occurrences of l
# consecutive pitch repetitions. We do not want the generator
# to repeat the same pitch many times in a row.

# Durations of Pitch Repetitions (DPR)
# For a specified duration d, measures the frequency of pitch repetitions that last
# at least d long in total. We do not want the generator to repeat
# the same pitch multiple times for a long time. For example,
# three whole notes of the same pitch in a row are worse than
# three triplets of the same pitch. We only consider repetitions
# of two or more notes.

# Tone Spans (TS)
# For a specified tone distance d, TS mea-
# sures the frequency of pitch changes that span more than d
# half-steps. Example: setting d = 12 counts the number of
# pitch leaps greater than an octave.

# Off-beat Recovery frequency (OR)
# Given an offset d, OR measures how frequently the model can recover back onto
# the beat after being forced to be off by d timesteps. For ex-
# ample, with a 48-timestep encoding for a bar, we run exper-
# iments with an offset of seven timesteps, which corresponds
# to no conventional beat position. We define recovery onto
# the beat as generating a note on a beat position correspond-
# ing to a multiple of an eighth note.

# Creativity Metrics

# We propose the following metrics to evaluate the creativity
# of the model.

# Rote Memorization frequencies (RM)
# Given a specified length l, RM measures how frequently the model copies
# note sequences of length l from the corpus.

# Pitch Variations (PV)
# PV measures how many distinct
# pitches the model plays within a sequence.

# Rhythm Variations (RV)
# RV measures how many distinct note durations the model plays within a sequence.

# Chord Harmony Metric

# We propose the following metric to evaluate how well the
# model interacts with the chord progression.

# Harmonic Consistency (HC)
# The harmonic consistency metric is based on the Impro-Visor (Keller 2018) note categorization,
# represented visually by coloration, which measures the frequency of black, green, blue, and red notes.
# Black notes are pitches that are part of the current chord,
# green notes (called ”color tones”) are tones sympathetic to
# the chord, blue notes are approach (by a half-step) tones to
# chord or color tones, and red notes are all other tones, which
# generally clash with the accompanying chord. The Impro-Visor vocabulary file defines these categories.
# We did not modify the standard file specifically for the current corpus.
# The HC metric turns on-chord and off-chord tones into
# more nuanced categories based on the surrounding context.
# This allows us to capture stylistic features such as approach
# tones, which are off-chord but resolve in the next note. Ide-
# ally, these frequencies should be close to the actual values
# for the training corpus

if __name__ == "__main__":
    print(1 - get_onset_xor_distance([1, 2, 3, 7, 8, 9], [4, 5, 6, 0, 10, 11]))
