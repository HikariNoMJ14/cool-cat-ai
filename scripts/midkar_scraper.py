from bs4 import BeautifulSoup
import os
import urllib.request
from urllib.error import HTTPError
from time import sleep

path = '/media/manu/DATA/Mac/Documents/University/Thesis/Renamed Raw Data/MidKar'

with open("sources/MidKar/MIDKAR.com Jazz MIDI Files U-Z.html", encoding="ISO-8859-1") as fp:
    soup = BeautifulSoup(fp, 'html.parser')

for a in soup.body.find_all('a'):
    if 'href' in a.attrs:
        fileurl = a.attrs['href']

        if '.mid' in fileurl:
            songname = " ".join(a.text.split()).replace("?", '').replace('/', ' - ')
            filename = fileurl.split('/')[-1]

            # print(songname, fileurl)

            try:
                urllib.request.urlretrieve(fileurl, os.path.join(path, f"{songname}.mid"))
            except HTTPError:
                print(songname, fileurl)

            sleep(0.5)


