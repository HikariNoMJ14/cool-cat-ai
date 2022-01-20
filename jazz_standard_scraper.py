from bs4 import BeautifulSoup
import os
import urllib.request
from time import sleep

path = './data/Raw Data/JazzStandards'
url = 'https://bhs.minor9.com/midi/jazzstandards/'

with open("./data/bhs.minor9.html", encoding="ISO-8859-1") as fp:
    soup = BeautifulSoup(fp, 'html.parser')

for table in soup.body.find_all('table'):
    for td in table.find_all('td'):
        a = td.find('a')

        if a:
            filename = a.attrs['href']
            fileurl = url + filename

            print(fileurl)

            try:
                urllib.request.urlretrieve(fileurl, os.path.join(path, f"{filename}"))
            except OSError as e:
                print(e)
                print(filename)

            sleep(0.5)