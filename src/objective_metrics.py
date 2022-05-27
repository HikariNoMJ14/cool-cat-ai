import json
import logging
import random, itertools

import scipy.stats
import numpy as np
import pandas as pd

import mingus.core.scales as scales
import mingus.core.notes as notes

from src.ezchord import Chord
from src.utils.constants import TICKS_PER_MEASURE, REST_VAL, PITCH_CLS

logger = logging.getLogger()


def compute_histogram_entropy(histogram):
    return scipy.stats.entropy(histogram) / np.log(2)


def get_pitch_histogram(melody, pitches=range(128), verbose=False):
    pitches = [x for x in melody['pitch'] if x in pitches]

    if not len(melody):
        if verbose:
            logger.info('The sequence contains no notes.')
        return None

    n_pich_cls = len(PITCH_CLS)

    pitches = pd.Series(pitches) % n_pich_cls

    histogram = pitches.value_counts(normalize=True)

    hist = np.zeros((n_pich_cls,))
    for i in range(n_pich_cls):
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

    # logger.info(a_onsets, b_onsets, np.sum(np.abs(a_onsets - b_onsets)))

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
                logger.info(
                    'No notes in this crop: {}~{} measures.'.format(start_measure, start_measure + window_size - 1))
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

valid_tempos = [3, 6, 12, 24, 48, 1, 2, 4, 8, 16, 0]
valids = []
for r1 in valid_tempos:
    for r2 in valid_tempos:
        rcalc = r1 + r2
        if rcalc not in valids:
            valids.append(rcalc)


def calculate_durations_training(seqs_dur, s):
    act_seq_durs = []

    for i in range(len(seqs_dur)):
        act_durs = []
        seq_dur = seqs_dur[i]
        prev_beatpos = s[i][0][1]
        for beatpos in seq_dur:
            dur = (beatpos - prev_beatpos + 48 - 1) % 48 + 1
            act_durs.append(dur)
            prev_beatpos = beatpos
        act_seq_durs.append(act_durs)
    return act_seq_durs


# Qualified Offset frequency (QR)
# QR measures the frequency of note durations within valid beat ratios of
# {1, 1/2, 1/4, 1/8, 1/16}, their dotted and triplet counter-
# parts, and any tied combination of two valid ratios. This
# generalizes beyond MuseGAN’s qualified note metric,
# which only measures the frequency of durations greater
# than a 32nd note.


def calculate_QO(melody):
    valid_count = 0.0
    total_count = 0.0

    sequence = list(melody['offset'])

    for offset in sequence:
        if offset in valids:
            valid_count += 1
        total_count += 1

    return valid_count / total_count


# Qualified Duration frequency (QR)
# QR measures the frequency of note durations within valid beat ratios of
# {1, 1/2, 1/4, 1/8, 1/16}, their dotted and triplet counter-
# parts, and any tied combination of two valid ratios. This
# generalizes beyond MuseGAN’s qualified note metric,
# which only measures the frequency of durations greater
# than a 32nd note.


def calculate_QD(melody):
    valid_count = 0.0
    total_count = 0.0

    sequence = list(melody['duration'])

    for duration in sequence:
        if duration in valids:
            valid_count += 1
        total_count += 1

    return valid_count / total_count


# Consecutive Pitch Repetitions (CPR)
# For a specified length l, CPR measures the frequency of occurrences of l
# consecutive pitch repetitions. We do not want the generator
# to repeat the same pitch many times in a row.


def calculate_CPR(melody, length):
    n_consecutive = 0.0
    total = 0.0

    pitches = list(melody['pitch'])

    repetition_count = 0
    index = 0
    sequence_pitch = -1
    while index <= len(pitches):
        if index == len(pitches):
            if repetition_count >= length:
                n_consecutive += 1
            total += 1
            break
        if sequence_pitch == -1:
            sequence_pitch = pitches[index]
            repetition_count = 1
        elif sequence_pitch != pitches[index]:
            if repetition_count >= length:
                n_consecutive += 1
            total += 1
            sequence_pitch = pitches[index]
            repetition_count = 1
        else:
            repetition_count += 1
        index += 1

    return n_consecutive / total


# Durations of Pitch Repetitions (DPR)
# For a specified duration d, measures the frequency of pitch repetitions that last
# at least d long in total. We do not want the generator to repeat
# the same pitch multiple times for a long time. For example,
# three whole notes of the same pitch in a row are worse than
# three triplets of the same pitch. We only consider repetitions
# of two or more notes.


def calculate_DPR(melody, duration):
    n_consecutive = 0.0
    total = 0.0

    pitches = list(melody['pitch'])
    durations = list(melody['duration'])

    repetition_count = 0
    index = 0
    sequence_duration = 0
    sequence_pitch = -1
    while index <= len(pitches):
        if index == len(pitches):
            if repetition_count > 1:
                if sequence_duration >= duration:
                    n_consecutive += 1
            total += 1
            break
        if sequence_pitch == -1:
            sequence_pitch = pitches[index]
            sequence_duration = durations[index]
            repetition_count = 1
        elif sequence_pitch != pitches[index]:
            if repetition_count > 1:
                if sequence_duration >= duration:
                    n_consecutive += 1
            total += 1
            sequence_pitch = pitches[index]
            sequence_duration = durations[index]
            repetition_count = 1
        else:
            sequence_duration += durations[index]
            repetition_count += 1
        index += 1

    return n_consecutive / total


# Tone Spans (TS)
# For a specified tone distance d, TS mea-
# sures the frequency of pitch changes that span more than d
# half-steps. Example: setting d = 12 counts the number of
# pitch leaps greater than an octave.


def calculate_TS(melody, distance):
    count = 0.0
    total = 0.0

    pitches = list(melody['pitch'])

    base_pitch = pitches[0]
    for i in range(len(pitches) - 1):
        i = i + 1
        if abs(pitches[i] - base_pitch) > distance and base_pitch != REST_VAL and pitches[i] != REST_VAL:
            count += 1
        total += 1
        base_pitch = pitches[i]

    return count / total


# Pitch Variations (PV)
# PV measures how many distinct
# pitches the model plays within a sequence.


def calculate_PV(melody, window_size):
    n_bars = melody['measure'].max() + 1

    pitch_variations = 0.0
    for start_measure in range(n_bars - window_size + 1):
        sequence = list(get_bars_crop(melody, start_measure, start_measure + window_size - 1)['pitch'] % 12)

        # logger.info(sequence)

        if len(sequence) > 0:
            diff_pitches = set([])
            for pitch in sequence:
                diff_pitches.add(pitch)
            pitch_variations += (float(len(diff_pitches)) / len(sequence))

    print(np.mean(pitch_variations), pitch_variations / (n_bars - window_size + 1))

    return np.mean(pitch_variations) if len(pitch_variations) > 0 else 0.0


# Rhythm Variations (RV)
# RV measures how many distinct note durations the model plays within a sequence.


def calculate_RV(melody, window_size):
    n_bars = melody['measure'].max() + 1

    rhythm_variation = []
    for start_measure in range(n_bars - window_size + 1):
        sequence = list(get_bars_crop(melody, start_measure, start_measure + window_size - 1)['duration'])

        if len(sequence) > 0:
            diff_durations = set([])
            for duration in sequence:
                diff_durations.add(duration)
            rhythm_variation.append(float(len(diff_durations)))

    # logger.info(rhythm_variation)

    return np.mean(rhythm_variation) if len(rhythm_variation) > 0 else 0.0


# Off-beat Recovery frequency (OR)
# Given an offset d, OR measures how frequently the model can recover back onto
# the beat after being forced to be off by d timesteps. For ex-
# ample, with a 48-timestep encoding for a bar, we run exper-
# iments with an offset of seven timesteps, which corresponds
# to no conventional beat position. We define recovery onto
# the beat as generating a note on a beat position correspond-
# ing to a multiple of an eighth note.


def calculate_OR(melody, l):
    valid_count = 0.0

    durations = melody['duration']

    for dur in durations[:l]:
        if dur % 6 == 0:
            valid_count += 1
            break

    return valid_count


# Rote Memorization frequencies (RM)
# Given a specified length l, RM measures how frequently the model copies
# note sequences of length l from the corpus.


def calculate_RM(melody, ns, ds, length=3, doDurs=False):
    ltuples = {}
    matchlist = []
    total = 0.0
    matches = 0.0

    pitches = melody['pitch']
    durations = melody['duration']

    for i in range(len(ns)):
        n = ns[i]
        d = ds[i]
        if len(n) < length:
            continue
        for j in range(len(n) - length + 1):
            if doDurs:
                seq = (tuple(n[j:j + length]), tuple(d[j:j + length]))
            else:
                seq = tuple(n[j:j + length])
            ltuples[seq] = True
    for i in range(len(pitches)):
        n = pitches[i]
        d = durations[i]
        if len(n) < length:
            continue
        for j in range(len(n) - length + 1):
            if doDurs:
                seq = (tuple(n[j:j + length]), tuple(d[j:j + length]))
            else:
                seq = tuple(n[j:j + length])
            try:
                result = ltuples[seq]
                if seq not in matchlist:
                    matchlist.append(seq)
                matches += 1
            except KeyError:
                pass
            total += 1

    logger.info('Num types of matches:', len(matchlist))

    return matches / total


# Harmonic Consistency (HC)
# The harmonic consistency metric is based on the Impro-Visor (Keller 2018) note categorization,
# represented visually by coloration, which measures the frequency of black, green, blue, and red notes.
# Black notes are pitches that are part of the current chord,
# green notes (called "color tones") are tones sympathetic to
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

def replace_enharmonic(pitch):
    if pitch == 'C#':
        return 'Db'
    if pitch == 'C##':
        return 'D'
    if pitch == 'D#':
        return 'Eb'
    if pitch == 'Fb':
        return 'E'
    if pitch == 'D##':
        return 'E'
    if pitch == 'E#':
        return 'F'
    if pitch == 'E##':
        return 'F#'
    if pitch == 'Gb':
        return 'F#'
    if pitch == 'F##':
        return 'G'
    if pitch == 'G#':
        return 'Ab'
    if pitch == 'G##':
        return 'B'
    if pitch == 'Bbb':
        return 'B'
    if pitch == 'A#':
        return 'Bb'
    if pitch == 'A##':
        return 'B'
    if pitch == 'Cb':
        return 'B'
    if pitch == 'B#':
        return 'C'
    return pitch


def get_chord_affinity(chord_name):
    chord_obj = Chord(chord_name)
    chord_notes = [notes.int_to_note(x) for x in np.array(chord_obj.getMIDI()) % 12]
    for idx, sn in enumerate(chord_notes):
        chord_notes[idx] = replace_enharmonic(sn)

    # logger.info("--- " + chord_name + " --- " + str(chord_notes))

    chord_affinity = {}

    for pc in PITCH_CLS:
        if pc in chord_notes:
            chord_affinity[pc] = 'ChT'
        else:
            chord_affinity[pc] = 'NaN'

    chord_root = chord_obj.root

    if chord_root == 'Gb':
        chord_root = 'F#'

    if chord_obj.mode.name == 'MAJ':
        scale = scales.Major(chord_root)
    elif chord_obj.mode.name == 'DOM':
        scale = scales.Mixolydian(chord_root)
    elif chord_obj.mode.name == 'MIN':
        scale = scales.Dorian(chord_root)
    elif chord_obj.mode.name == 'DIM':
        scale = scales.HarmonicMinor(chord_root)
    elif chord_obj.mode.name == 'AUG':
        scale = scales.Major(chord_root)
    elif chord_obj.mode.name == 'SUS':
        scale = scales.Major(chord_root)
    else:
        logger.info('Unknown mode: ' + chord_name + ' - ' + chord_obj.mode.name)

    scale_notes = scale.ascending()

    if chord_obj.mode.name == 'DIM':
        scale_notes[4] = notes.int_to_note((notes.note_to_int(scale_notes[4]) - 1) % 12)
        scale_notes.insert(6, notes.int_to_note((notes.note_to_int(scale_notes[5]) + 1) % 12))

    if 'b9' in chord_name:
        scale_notes[1] = notes.int_to_note((notes.note_to_int(scale_notes[1]) - 1) % 12)
    if 'b11' in chord_name:
        scale_notes[3] = notes.int_to_note((notes.note_to_int(scale_notes[3]) - 1) % 12)
    if 'b5' in chord_name:
        scale_notes[4] = notes.int_to_note((notes.note_to_int(scale_notes[4]) - 1) % 12)
    if 'b13' in chord_name:
        scale_notes[5] = notes.int_to_note((notes.note_to_int(scale_notes[5]) - 1) % 12)
    if '#9' in chord_name:
        scale_notes[1] = notes.int_to_note((notes.note_to_int(scale_notes[1]) + 1) % 12)
    if '#11' in chord_name:
        scale_notes[3] = notes.int_to_note((notes.note_to_int(scale_notes[3]) + 1) % 12)
    if 'sus' in chord_name:
        scale_notes[3] = notes.int_to_note((notes.note_to_int(scale_notes[3]) + 1) % 12)
    if '#5' in chord_name:
        scale_notes[4] = notes.int_to_note((notes.note_to_int(scale_notes[4]) + 1) % 12)
    if '+' in chord_name:
        scale_notes[4] = notes.int_to_note((notes.note_to_int(scale_notes[4]) + 1) % 12)
    if '#13' in chord_name:
        scale_notes[5] = notes.int_to_note((notes.note_to_int(scale_notes[5]) + 1) % 12)

    # logger.info(chord_obj.mode.name, scale_notes)
    # logger.info(chord_affinity)

    for idx, sn in enumerate(scale_notes):
        scale_notes[idx] = replace_enharmonic(sn)

    for scale_note in scale_notes:
        if chord_affinity[scale_note] == "NaN":
            chord_affinity[scale_note] = 'CoT'

    chot = 0
    colt = 0
    nant = 0

    for pc in PITCH_CLS:
        if chord_affinity[pc] == 'ChT':
            chot += 1
        if chord_affinity[pc] == 'CoT':
            colt += 1
        if chord_affinity[pc] == 'NaN':
            nant += 1

    # logger.info(chot, colt, nant)

    return chord_affinity


def calculate_HC(melody):
    harmonic_consistency = []

    pitches = list(melody['pitch'])
    chord_names = list(melody['chord_name'])
    durations = list(melody['duration'])

    for idx, pitch in enumerate(pitches):
        score = 0

        cur_chord = chord_names[idx]
        pitch = replace_enharmonic(notes.int_to_note(pitch % 12))

        cur_chord_affinity = get_chord_affinity(cur_chord)

        if cur_chord_affinity[pitch] == 'ChT':
            score = 1.0
        elif cur_chord_affinity[pitch] == 'CoT':
            score = .66
        elif cur_chord_affinity[pitch] == 'NaN':
            score = .0

            if len(chord_names) > idx + 1 and chord_names[idx + 1] != cur_chord:
                next_chord = chord_names[idx + 1]
                next_chord_affinity = get_chord_affinity(next_chord)

                next_pitch = replace_enharmonic(notes.int_to_note((notes.note_to_int(pitch) + 1) % 12))
                previous_pitch = replace_enharmonic(notes.int_to_note((notes.note_to_int(pitch) - 1) % 12))

                if durations[idx] <= 12 and \
                        (next_chord_affinity[next_pitch] == 'ChT' or next_chord_affinity[previous_pitch] == 'ChT'):
                    score = 0.33

        harmonic_consistency.append(score)

    return harmonic_consistency


def calculate_silence_ratio(melody_data):
    n_measures = melody_data['measure'].unique()[-1]
    c = np.zeros(n_measures)
    previous = None

    for measure in range(n_measures):
        current = melody_data[melody_data['measure'] == measure]

        if len(current) > 0:
            c[measure] = 1

        if previous is not None:
            end_offset = (previous['offset'] + previous['duration']) / 48
            long_notes = end_offset[end_offset >= 1]

            if len(long_notes) > 0:
                c[measure] = 1
                longer_notes = end_offset[end_offset >= 2]

                if len(longer_notes) > 0:
                    if len(c) > measure + 1:
                        c[measure + 1] = 1
                    longest_notes = end_offset[end_offset >= 2]

                    if len(longest_notes) > 0:
                        if len(c) > measure + 2:
                            c[measure + 2] = 1

        previous = current

    return 1 - c.mean()


def get_sequences_onefile(allpath, MAX_SEQ_DUR_LENGTH):
    MIDI_MAX = 108
    MIDI_MIN = 36
    REST_VAL = MIDI_MAX - MIDI_MIN + 1
    logger.info(allpath)
    with open(allpath, 'r') as infile:
        data = json.load(infile)
    mt_list = data["transposed_seqs_skip"]
    ct_list = data["full_chords"]
    # logger.info(ct_list[5][0][0])
    all_seqs = []
    maxseqdur = 0
    for lindex in range(len(mt_list)):
        mt = mt_list[lindex]
        ct = ct_list[lindex]
        for tindex in range(len(mt)):
            m = mt[tindex]
            c = ct[tindex]
            for seq in m:
                seqStart = -1
                for i in range(len(seq)):
                    if seq[i][0] != REST_VAL:
                        seqStart = i
                        break
                if seqStart == -1:
                    continue
                seqEnd = -1
                for i in range(len(seq)):
                    if seq[len(seq) - 1 - i][0] != REST_VAL:
                        seqEnd = len(seq) - i
                        break
                if seqEnd != -1:
                    seq = seq[seqStart:seqEnd]
                else:
                    seq = seq[seqStart:]
                startseqsize = 4
                if len(seq) <= startseqsize:  # or len(seq) > MAX_SEQ_LENGTH:
                    continue
                startseq = seq[:startseqsize]
                startseq.reverse()
                seq = seq[startseqsize:]
                noteseq = []
                durseq = []
                chordseq = []
                lowseq = []
                highseq = []
                prevpos = 0
                spseq = []  # noteval, dur, beatpos, chordkey, otherdur
                for pitch, beatpos, index in startseq:
                    dur = (beatpos - prevpos + 48 - 1) % 48  # actual durs are 1-48, but one-hotted is 0-47
                    spseq.append([pitch, beatpos, dur, c[index][0], dur])
                    prevpos = beatpos
                    # for _ in range(dur):
                    #    chordseq.append(c[index])
                seqdur = 0
                for pitch, beatpos, index in seq:
                    noteseq.append(pitch)
                    durseq.append(beatpos)
                    dur = (beatpos - prevpos + 48 - 1) % 48 + 1  # -1 +1 for octaves
                    prevpos = beatpos
                    seqdur += dur
                    for _ in range(dur):
                        chordseq.append(c[index])
                    if pitch == REST_VAL:
                        lowseq.append(0.0)
                        highseq.append(0.0)
                    else:
                        lowseq.append(float(pitch / float(MIDI_MAX - MIDI_MIN)))
                        highseq.append(1.0 - float(pitch / float(MIDI_MAX - MIDI_MIN)))
                if seqdur > MAX_SEQ_DUR_LENGTH:
                    continue
                all_seqs.append([noteseq, durseq, chordseq, lowseq, highseq, spseq])
                if seqdur > maxseqdur:
                    maxseqdur = seqdur
                    prevpos = 0
                    durs = []
                    for pitch, beatpos, index in seq:
                        durs.append((beatpos - prevpos + 48 - 1) % 48 + 1)
                        prevpos = beatpos
                    logger.info(durs)
                    logger.info(sum(durs))

    logger.info(maxseqdur)
    all_seqs.sort(key=lambda x: len(x[0]), reverse=False)
    noteseqs = [x[0] for x in all_seqs]
    durseqs = [x[1] for x in all_seqs]
    chordseqs = [x[2] for x in all_seqs]
    lows = [x[3] for x in all_seqs]
    highs = [x[4] for x in all_seqs]
    spseqs = [x[5] for x in all_seqs]
    logger.info("Number of sequences: ", len(noteseqs))
    return noteseqs, durseqs, chordseqs, lows, highs, spseqs


if __name__ == "__main__":
    filepaths = [
        # '../data/split_melody_data/Real Book/All Of Me_original.csv',
        # '../data/split_melody_data/Jazz-Midi/All Of Me_1.csv'
        # '../data/split_melody_data/Doug McKenzie/Alone Together_1.csv'
        '../data/finalised/csv/Real Book/Falling Grace -o-.csv'
    ]

    df = pd.read_csv(filepaths[0], index_col=0)
    # df['chord_notes'] = df['chord_notes'].apply(ast.literal_eval)
    #
    # df2 = pd.read_csv(filepaths[1], index_col=0)
    # df2['chord_notes'] = df2['chord_notes'].apply(ast.literal_eval)

    calculate_HC(df)

    # r1 = calculate_QO(df)
    # r2 = calculate_QD(df)
    # r3 = calculate_CPR(df, 2)
    # r4 = calculate_DPR(df, 12)
    # r5 = calculate_TS(df, 12)
    # r4 = calculate_OR(df, 6)

    # MAX_SEQ_DUR_LENGTH = 48 * 4
    # fname = "/home/nic/sequence_gan/parsed_leadsheets_bricked_all2_dur/pitchexpert_onehot_features.json"
    # n, d, _, _, _, starts = get_sequences_onefile(fname, MAX_SEQ_DUR_LENGTH)
    # act_d = calculate_durations_training(d, starts)
    # for l in [3, 4, 5, 6]:
    #     logger.info('RM', calculate_RM(df, n, act_d, length=l, doDurs=False), 'l', l)

    # r6 = calculate_PV(df, 4)
    # r7 = calculate_RV(df, 4)
    #
    # r8 = calculate_PV(df2, 4)
    # r9 = calculate_RV(df2, 4)

    # import seaborn as sns
    # import matplotlib.pyplot as plt
    #
    # sns.countplot(df['duration'])
    # sns.countplot(df2['duration'])
    # plt.show()

    # logger.info(r1)
    # logger.info(r2)
    # logger.info(r3)
    # logger.info(r4)
    # logger.info(r5)
    # logger.info(r6)
    # logger.info(r7)

    df = pd.DataFrame(data=[
        # {'pitch': 72, 'duration': 21, 'offset': 21, 'measure': 0, 'chord_notes': []},
        # {'pitch': 72, 'duration': 21, 'offset': 21, 'measure': 0, 'chord_notes': []},
        # {'pitch': 72, 'duration': 12, 'offset': 21, 'measure': 0, 'chord_notes': []},
        # {'pitch': 72, 'duration': 12, 'offset': 21, 'measure': 0, 'chord_notes': []},
        # {'pitch': 72, 'duration': 21, 'offset': 21, 'measure': 1, 'chord_notes': []},
        # {'pitch': 72, 'duration': 21, 'offset': 21, 'measure': 1, 'chord_notes': []},
        # {'pitch': 72, 'duration': 12, 'offset': 21, 'measure': 1, 'chord_notes': []},
        # {'pitch': 72, 'duration': 12, 'offset': 21, 'measure': 1, 'chord_notes': []},
        # {'pitch': 72, 'duration': 21, 'offset': 21, 'measure': 2, 'chord_notes': []},
        # {'pitch': 72, 'duration': 21, 'offset': 21, 'measure': 2, 'chord_notes': []},
        # {'pitch': 72, 'duration': 12, 'offset': 21, 'measure': 2, 'chord_notes': []},
        # {'pitch': 72, 'duration': 12, 'offset': 21, 'measure': 2, 'chord_notes': []},
        # {'pitch': 72, 'duration': 21, 'offset': 21, 'measure': 3, 'chord_notes': []},
        # {'pitch': 72, 'duration': 21, 'offset': 21, 'measure': 3, 'chord_notes': []},
        # {'pitch': 72, 'duration': 12, 'offset': 21, 'measure': 3, 'chord_notes': []},
        # {'pitch': 72, 'duration': 12, 'offset': 21, 'measure': 3, 'chord_notes': []},

        {'pitch': 74, 'duration': 21, 'offset': 21, 'measure': 0, 'chord_name': 'Dm7'},
        {'pitch': 72, 'duration': 21, 'offset': 21, 'measure': 0, 'chord_name': 'Dm7'},
        {'pitch': 73, 'duration': 12, 'offset': 21, 'measure': 0, 'chord_name': 'A7b9'},
        {'pitch': 76, 'duration': 12, 'offset': 21, 'measure': 0, 'chord_name': 'A7b9'},
        {'pitch': 78, 'duration': 1, 'offset': 21, 'measure': 1, 'chord_name': 'D7'},
        {'pitch': 79, 'duration': 1, 'offset': 21, 'measure': 1, 'chord_name': 'D7'},
        {'pitch': 77, 'duration': 1, 'offset': 21, 'measure': 1, 'chord_name': 'G7#5'},
        {'pitch': 75, 'duration': 1, 'offset': 21, 'measure': 1, 'chord_name': 'G7#5'},
        {'pitch': 76, 'duration': 1, 'offset': 21, 'measure': 1, 'chord_name': 'Cmaj9'},
        {'pitch': 74, 'duration': 1, 'offset': 21, 'measure': 2, 'chord_name': 'Cmaj9'},
        {'pitch': 69, 'duration': 1, 'offset': 21, 'measure': 2, 'chord_name': 'Cmaj9'},
        {'pitch': 70, 'duration': 1, 'offset': 21, 'measure': 2, 'chord_name': 'Cmaj9'},
        {'pitch': 75, 'duration': 1, 'offset': 21, 'measure': 2, 'chord_name': 'Fmaj7'},
        {'pitch': 75, 'duration': 1, 'offset': 21, 'measure': 2, 'chord_name': 'Fmaj7'},
        {'pitch': 70, 'duration': 21, 'offset': 21, 'measure': 3, 'chord_name': 'Em7b5'},
        {'pitch': 69, 'duration': 21, 'offset': 21, 'measure': 3, 'chord_name': 'Em7b5'},
        {'pitch': 71, 'duration': 12, 'offset': 21, 'measure': 3, 'chord_name': 'A7'},
        {'pitch': 77, 'duration': 12, 'offset': 21, 'measure': 3, 'chord_name': 'A7'},

        # {'pitch': 72, 'duration': 21, 'offset': 21, 'measure': 0, 'chord_notes': []},
        # {'pitch': 72, 'duration': 21, 'offset': 21, 'measure': 0, 'chord_notes': []},
        # {'pitch': 70, 'duration': 12, 'offset': 21, 'measure': 0, 'chord_notes': []},
        # {'pitch': 69, 'duration': 12, 'offset': 21, 'measure': 0, 'chord_notes': []},
        # {'pitch': 76, 'duration': 1, 'offset': 21, 'measure': 1, 'chord_notes': []},
        # {'pitch': 75, 'duration': 1, 'offset': 21, 'measure': 1, 'chord_notes': []},
        # {'pitch': 74, 'duration': 1, 'offset': 21, 'measure': 1, 'chord_notes': []},
        # {'pitch': 73, 'duration': 1, 'offset': 21, 'measure': 1, 'chord_notes': []},
        # {'pitch': 72, 'duration': 1, 'offset': 21, 'measure': 1, 'chord_notes': []},
        # {'pitch': 75, 'duration': 1, 'offset': 21, 'measure': 2, 'chord_notes': []},
        # {'pitch': 75, 'duration': 1, 'offset': 21, 'measure': 2, 'chord_notes': []},
        # {'pitch': 75, 'duration': 1, 'offset': 21, 'measure': 2, 'chord_notes': []},
        # {'pitch': 75, 'duration': 1, 'offset': 21, 'measure': 2, 'chord_notes': []},
        # {'pitch': 75, 'duration': 1, 'offset': 21, 'measure': 2, 'chord_notes': []},
        # {'pitch': 71, 'duration': 21, 'offset': 21, 'measure': 3, 'chord_notes': []},
        # {'pitch': 78, 'duration': 21, 'offset': 21, 'measure': 3, 'chord_notes': []},
        # {'pitch': 71, 'duration': 12, 'offset': 21, 'measure': 3, 'chord_notes': []},
        # {'pitch': 77, 'duration': 12, 'offset': 21, 'measure': 3, 'chord_notes': []},
    ])

    # logger.info(sorted(valids))
    # logger.info(calculate_QO(df))
    # logger.info(calculate_QD(df))
    # logger.info(calculate_CPR(df, 2))
    # logger.info(calculate_DPR(df, 12))
    # logger.info(calculate_TS(df, 12))
    # logger.info(calculate_PV(df, 4))
    # logger.info(calculate_RV(df, 1))
    # logger.info(calculate_RV(df, 4))
    logger.info(calculate_HC(df))
