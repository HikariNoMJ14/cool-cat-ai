import pretty_midi
import pandas as pd
import numpy as np


def midi_to_notes(midi_file: str) -> pd.DataFrame:
    pm = pretty_midi.PrettyMIDI(midi_file)
    instrument = pm.instruments[0]
    notes = pretty_midi.collections.defaultdict(list)

    # Sort the notes by start time
    sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
    prev_start = sorted_notes[0].start
    init_time = sorted_notes[0].start

    for note in sorted_notes:
        start = note.start - init_time
        end = note.end - init_time
        notes['pitch'].append(note.pitch)
        notes['start'].append(start)
        notes['end'].append(end)
        notes['velocity'].append(note.velocity)
        notes['step'].append(start - prev_start)
        notes['duration'].append(end - start)
        prev_start = start
    notes["step"][0] = 0

    return pd.DataFrame({name: np.array(value) for name, value in notes.items()})


def notes_to_midi(notes: pd.DataFrame,
                  out_file: str,
                  instrument_name: str = "Acoustic Grand Piano") -> pretty_midi.PrettyMIDI:

    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program(
            instrument_name))

    for i, note in notes.iterrows():
        note = pretty_midi.Note(
            velocity=int(note["velocity"]),
            pitch=int(note["pitch"]),
            start=note["start"],
            end=note["end"],
        )
        instrument.notes.append(note)

    pm.instruments.append(instrument)
    pm.write(out_file)
    return pm
