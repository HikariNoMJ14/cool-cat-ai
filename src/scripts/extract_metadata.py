import os
import sys

import pandas as pd

sys.path.append(os.path.join('..', '..'))

metadata = pd.read_csv('../../data/finalised/metadata.csv', index_col=0)
wei_old = metadata[metadata['source'] == 'Weimar DB']
wei_old['performer'] = wei_old['filename'].str.split(' - ').apply(lambda x: x[0])
wei_old = wei_old.sort_values(['song_name', 'performer']).reset_index(drop=True)
nowei = metadata[metadata['source'] != 'Weimar DB'].reset_index(drop=True)


solo_info = pd.read_csv('/home/manu/Downloads/weimar_db_solo_info.csv')

wei_new = solo_info[['performer', 'title', 'avgtempo']].sort_values('title').reset_index(drop=True)
wei_new = wei_new.iloc[3:]
wei_new = wei_new.drop(range(5, 11))
wei_new = wei_new.drop(range(13, 16))
wei_new = wei_new.drop(range(22, 24))
wei_new = wei_new.drop(25)
wei_new = wei_new.drop(29)
wei_new = wei_new.drop(range(32, 37))
wei_new = wei_new.drop(range(38, 43))
wei_new = wei_new.drop(range(45, 57))
wei_new = wei_new.drop(range(60, 62))
wei_new = wei_new.drop(range(63, 93))
wei_new = wei_new.drop(range(94, 109))
wei_new = wei_new.drop(range(110, 119))
wei_new = wei_new.drop(range(120, 138))
wei_new = wei_new.drop(range(141, 142))
wei_new = wei_new.drop(range(143, 144))
wei_new = wei_new.drop(range(145, 156))
wei_new = wei_new.drop(range(158, 161))
wei_new = wei_new.drop(164)
wei_new = wei_new.drop(167)
wei_new = wei_new.drop(range(170, 174))
wei_new = wei_new.drop(range(175, 186))
wei_new = wei_new.drop(range(187, 192))
wei_new = wei_new.drop(193)
wei_new = wei_new.drop(range(195, 197))
wei_new = wei_new.drop(198)
wei_new = wei_new.drop(200)
wei_new = wei_new.drop(range(205, 211))
wei_new = wei_new.drop(range(214, 218))
wei_new = wei_new.drop(range(219, 231))
wei_new = wei_new.drop(232)
wei_new = wei_new.drop(range(236, 247))
wei_new = wei_new.drop(range(248, 249))
wei_new = wei_new.drop(range(251, 253))
wei_new = wei_new.drop(256)
wei_new = wei_new.drop(260)
wei_new = wei_new.drop(range(263, 274))
wei_new = wei_new.drop(range(278, 280))
wei_new = wei_new.drop(range(281, 286))
wei_new = wei_new.drop(range(289, 295))
wei_new = wei_new.drop(299)
wei_new = wei_new.drop(range(303, 306))
wei_new = wei_new.drop(range(307, 315))
wei_new = wei_new.drop(range(316, 346))
wei_new = wei_new.drop(range(347, 358))
wei_new = wei_new.drop(range(361, 363))
wei_new = wei_new.drop(365)
wei_new = wei_new.drop(range(367, 369))
wei_new = wei_new.drop(range(371, 375))
wei_new = wei_new.drop(range(377, 386))
wei_new = wei_new.drop(range(387, 402))
wei_new = wei_new.drop(range(403, 410))
wei_new = wei_new.drop(range(413, 437))
wei_new = wei_new.drop(range(438, 449))
wei_new = wei_new.drop(range(450, 456))

wei_new = wei_new.sort_values(['title', 'performer']).reset_index(drop=True)
wei_new['tempo'] = wei_new['avgtempo'].apply(round)
wei_new['tempo_idx'] = wei_new['tempo']

wei_tot = pd.concat([wei_old.drop(['tempo', 'tempo_idx'], axis=1),
                     wei_new[['tempo', 'tempo_idx']]], axis=1)

tot = pd.concat([nowei, wei_tot], axis=0)

tempo_mapping = {}
for k,v in enumerate(set(tot['tempo'].sort_values().unique())):
    tempo_mapping[v] = k

tot['tempo_idx'] = tot['tempo'].apply(lambda x: tempo_mapping[x])

tot.to_csv('../../data/finalised/metadata_v2.csv')

