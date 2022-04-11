import re
import pandas as pd

def get_offset(row):
    beat = int(row['beat'])
    tat = int(row['tatum'])
    sub = int(row['subtatum'])
    div = int(row['division'])

    tpb = 12

    if sub != 0:
        print('Sub!!!!')

    offset = int((beat - 1) * tpb + ((tat - 1) / div) * tpb)

    return offset


def create_note_info(info):
    ni = info[~info['pitch'].isnull()].reset_index(drop=True)

    time_signature = ni['period'].unique()[0]
    tpb = 12

    ni['pitch'] = ni['pitch'].apply(int)
    ni['measure'] = ni['bar']
    ni['offset'] = ni.apply(get_offset, axis=1)

    if ni['offset'].max() >= 12 * time_signature:
        print('Offset bigger than 47')

    ni['quant_ticks'] = ni['offset'] + ni['bar'] * tpb * time_signature
    ni['duration'] = (ni['duration'] * (ni['avgtempo'] / 60) * tpb).apply(round)

    ni['raw_ticks'] = ni['quant_ticks']
    ni['raw_duration'] = ni['duration']
    ni['quant_duration'] = ni['duration']

    ni = ni[ni['duration'] > 0]

    return ni[['pitch', 'raw_ticks', 'quant_ticks', 'raw_duration', 'quant_duration', 'offset', 'measure']]


def flatten_wdb_chord_progression(performer, song_name, base_folder='../..'):
    info = pd.read_csv(base_folder + '/data/sources/WeimarDB/weimar_db_melody_info.csv')

    rows = info[(info['performer'] == performer) & (info['title'] == song_name)]

    for i, row in rows.drop_duplicates('chord_changes').iterrows():
        c = row['chord_changes']

        parts = c.split('\n')
        song_structure = {
            'sections': [],
            'progression': {},
            'key': row['key']
        }

        for part in parts:
            pn = part.split(':')[0]

            cp = part.split(':')[1].strip().replace('||', '').split('|')

            chord_progression = []

            for m in cp:
                m = m.replace('NC', ' ')
                cs = re.split(r'([ ,A-Z][a-z,#,0-9,-]*)', m)

                for c in cs:
                    if c != '' and c != " ":
                        chord = c

                    if c != '':
                        chord_progression.append(chord)

            song_structure['progression'][pn] = chord_progression

        filter_info = info[info['melid'] == row['melid']]

        sections = list(filter_info['form'].drop(['I1'], errors='ignore').dropna())
        try:
            sections.remove('I1')
        except:
            pass
        try:
            sections.remove('I1')
        except:
            pass
        try:
            sections.remove('I1')
        except:
            pass
        try:
            sections.remove('I')
        except:
            pass
        try:
            sections.remove('I')
        except:
            pass

        song_structure['sections'] = sections

        linear_chord_progression = []
        for section in song_structure['sections']:
            linear_chord_progression += song_structure['progression'][section.replace('*', '')]

        # print(song_structure['key'])
        # print('------')
        print(song_structure['sections'])
        # print('------')
        # for k, p in song_structure['progression'].items():
        #     print(k)
        #     print(p)
        #     print('------')

        return linear_chord_progression
