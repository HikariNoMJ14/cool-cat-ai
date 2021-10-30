from bs4 import BeautifulSoup
import os

path = '/media/manu/DATA/Mac/Documents/University/Thesis/Renamed Raw Data/JazzPage'

with open("./data/TheJazzPage.html", encoding="ISO-8859-1") as fp:
    soup = BeautifulSoup(fp, 'html.parser')

for table in soup.body.find_all('table'):
    for td in table.find_all('td', class_='darkleft'):
        a = td.find('a')

        filename = a.attrs['href'].replace('sounds/', '')

        songname = " ".join(a.text.split())

        # print(songname, filename)

        try:
            if os.path.exists(os.path.join(path, filename)):
                os.rename(os.path.join(path, filename), f'{os.path.join(path, songname)}.mid')
        except OSError:
            print(songname)