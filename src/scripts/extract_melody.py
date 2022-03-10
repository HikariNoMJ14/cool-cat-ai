import pretty_midi as pm
from mido import MidiFile
import os
from glob import glob
import pandas as pd
import re
import copy

from utils import filter_instruments_new, get_melody_tracks


def get_channels(m):
    channels = set({})
    for t in m.tracks:
        for m in t:
            if m.type == 'note_on':
                channels.add(m.channel)

    return channels


def get_programs_mido(m, channels):
    programs = []

    for c in channels:
        programs.append(tuple([0, c]))

    for t in m.tracks:
        for m in t:
            if m.type == 'program_change':
                programs.append(tuple([m.program, m.channel]))

    return programs


def get_programs_pm(m):
    programs = []

    for i in m.instruments:
        programs.append(tuple([i.program, i.is_drum]))

    return programs


def extract(file):
    status = 'unprocessed'
    error = None

    source = os.path.basename(os.path.dirname(file))

    single_melody = False
    multiple_melody = False
    solo_channel = False
    multi_channel = False
    disagreement = False
    single_candidate = False

    melody_tracks = []
    filtered_instr = []
    melody_track = None

    song_name = os.path.basename(file).replace('.* - ', '').replace('.mid', '')
    song_name = re.sub(r'\([0-9]*\)', '', song_name)

    melody_track_name = None

    channels = None
    candidate_names = []
    n_channels = None
    n_instr = None
    mido_p = None
    pm_p = None
    midi_type = None

    instruments = []

    try:
        mido_m = MidiFile(file, clip=True)
        midi_type = mido_m.type

        channels = get_channels(mido_m)
        n_channels = len(channels)
        mido_p = get_programs_mido(mido_m, channels)

    except Exception as e:
        print(file, e)
        error = e

    try:
        pm_m = pm.PrettyMIDI(file)
        pm_p = get_programs_pm(pm_m)

        instruments = pm_m.instruments

        for p, _ in pm_p:
            if p not in [m[0] for m in mido_p]:
                error = 'program mismatch'

        melody_tracks = get_melody_tracks(pm_m)
        n_instr = len(pm_m.instruments)

        filtered_instr = filter_instruments_new(pm_m)
        candidate_names = [i.name for i in filtered_instr]

        if n_instr == 1:
            solo_channel = True
            melody_track = pm_m.instruments[0]

        if len(melody_tracks) == 1:
            melody_track = melody_tracks[0]
            single_melody = True
            single_candidate = True

        if len(melody_tracks) > 1:
            multiple_melody = True

        if len(filtered_instr) == 1:
            melody_track = filtered_instr[0]
            single_candidate = True

        if len(filtered_instr) > 1:
            multi_channel = True

        if n_channels != n_instr:
            disagreement = True

        if single_melody or solo_channel or single_candidate:
            status = 'processed'

            out_path = os.path.join('..', '..', 'data', 'Complete Examples Melodies Auto', f'v{version}')

        if status == 'unprocessed':
            out_path = os.path.join('..', '..', 'data', 'Complete Examples Melodies Random', f'v{version}')

            melody_track = filtered_instr[0]

        if melody_track:
            melody_track_name = melody_track.name

        if not os.path.exists(os.path.join(out_path, source)):
            os.makedirs(os.path.join(out_path, source))

        out_filename = os.path.join(out_path, source, os.path.basename(file))

        pm_melody = copy.deepcopy(pm_m)
        pm_melody.instruments = [melody_track]

        pm_melody.write(out_filename)

    except Exception as e:
        print(file, e)
        error = e

    return {
        'filepath': os.path.join(source, os.path.basename(file)),
        'source': source,
        'filename': os.path.basename(file),
        'song_name': song_name,
        'midi_type': midi_type,
        'channels': channels,
        'n_channels': n_channels,
        'instruments': instruments,
        'n_instruments': n_instr,
        'programs_mido': mido_p,
        'programs_pm': pm_p,
        'melody_tracks': melody_tracks,
        'melody_track': melody_track,
        'melody_track_name': melody_track_name,
        'candidates': filtered_instr,
        'candidate_names': candidate_names,
        'single_melody': single_melody,
        'multiple_melody': multiple_melody,
        'solo_channel': solo_channel,
        'single_candidate': single_candidate,
        'multi_channel': multi_channel,
        'disagreement': disagreement,
        'status': status,
        'error': error
    }


if __name__ == "__main__":
    version = '1.2'

    output_csv = f'../../data/melody_extraction/v{version}.csv'

    if os.path.exists(output_csv):
        print('File already exists!')
        exit(1)

    songs_list = []

    folder = f"../../data/Complete Examples/v{version}"

    files = [y for x in os.walk(folder) for y in glob(os.path.join(x[0], '*.mid'))]

    for file in files[:]:
        song = extract(file)
        songs_list.append(song)

    songs = pd.DataFrame().from_dict(songs_list)

    songs.to_csv(output_csv)
