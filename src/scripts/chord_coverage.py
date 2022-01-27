import os
from glob import glob
import re
import pandas as pd

melody_folder = "/media/manu/DATA/Mac/Documents/University/Thesis/Complete Examples Melodies"
melody_files = [os.path.basename(y).replace('.mid', '')
                for x in os.walk(melody_folder) for y in glob(os.path.join(x[0], '*.mid'))]

irb_folder = "./data/iRb_v1-0"
irb_files = [os.path.basename(y).replace('.jazz', '')
             for x in os.walk(irb_folder) for y in glob(os.path.join(x[0], '*.jazz'))]

jaah_folder = "./data/JAAH-v0.1/MTG-JAAH-7686b91/annotations"
jaah_files = [os.path.basename(y).replace('.json', '')
              for x in os.walk(jaah_folder) for y in glob(os.path.join(x[0], '*.json'))]

open_folder = "./openbook-master/src/openbook"
open_files = [os.path.basename(y).replace('.ly.mako', '')
              for x in os.walk(open_folder) for y in glob(os.path.join(x[0], '*.ly.mako'))]

wdb = list(pd.read_csv('sources/WeimarDB/wjazzd_composition_info.csv')['title'])

all_chord_files = set([])
all_chord_files.union(set(irb_files))
all_chord_files.union(set(jaah_files))
all_chord_files.union(set(open_files))
all_chord_files.union(set(wdb))

songnames = set([])
found = set([])

irb_n = 0
jaah_n = 0
open_n = 0
wdb_n = 0

for songname in melody_files:
    songname = "".join(songname.split(' - ')[-1])
    songname = re.sub('\(.*\)', '', songname).strip()
    songnames.add(songname)

    u_songname = songname.lower() \
        .replace(',', '') \
        .replace('\'', '') \
        .replace('-', '') \
        .replace(',', '')

    ut_songname = u_songname.replace(' ', '')

    uu_songname = u_songname.replace(' ', '_')

    # print(uu_songname)

    if uu_songname in jaah_files:
        # print('jaah', songname)
        found.add(songname)
        jaah_n += 1
    if ut_songname in irb_files:
        # print('irb', songname)
        found.add(songname)
        irb_n += 1
    if songname in wdb:
        # print('wdb', songname)
        found.add(songname)
        wdb_n += 1
    if uu_songname in open_files:
        # print('open', songname)
        found.add(songname)
        open_n += 1

left = songnames.difference(found)
print(len(songnames), len(found), len(left))
print('----------------------')

for l in sorted(list(set(left))):
    print(l)

print(f'jaah: {jaah_n}, irb: {irb_n}, open: {open_n}, wdb: {wdb_n}')
