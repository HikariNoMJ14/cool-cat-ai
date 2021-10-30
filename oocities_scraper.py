from bs4 import BeautifulSoup
import os
import urllib.request
from urllib.error import HTTPError
from time import sleep

path = '/media/manu/DATA/Mac/Documents/University/Thesis/Renamed Raw Data/Oocities'

with open("./data/justjaz3.htm", encoding="ISO-8859-1") as fp:
    soup = BeautifulSoup(fp, 'html.parser')

for a in soup.body.find_all('a'):
    if 'href' in a.attrs:
        fileurl = os.path.join("http://www.oocities.org/bourbonstreet/1114", a.attrs['href'])

        if '.mid' in fileurl:
            songname = " ".join(a.text.split()).replace("?", '').replace('/', ' - ')
            filename = fileurl.split('/')[-1]

            # print(songname, fileurl)

            try:
                urllib.request.urlretrieve(fileurl, os.path.join(path, f"{songname}.mid"))
            except HTTPError:
                print(songname, fileurl)

            sleep(0.5)