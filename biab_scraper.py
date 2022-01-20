from bs4 import BeautifulSoup
import os
import urllib.request
from time import sleep

path = './data/Raw Data/Biab'
url = 'https://bhs.minor9.com/'


pages = ['%23', "A", "B", "C"]

for page in pages:
    pageurl = url + '/biab/jazz/' + page
    urllib.request.urlretrieve(fileurl, os.path.join(path, f"biab{page}.html"))

# with open("./data/biab#.html", encoding="ISO-8859-1") as fp:
#     soup = BeautifulSoup(fp, 'html.parser')
#
#
# for table in soup.body.find_all('table'):
#     for td in table.find_all('td', class_='fb-n'):
#         a = td.find('a')
#
#         if a:
#             filename = a.attrs['href']
#             fileurl = url + filename
#
#             print(fileurl)
#
#             try:
#                 urllib.request.urlretrieve(fileurl, os.path.join(path, f"{os.path.basename(filename)}"))
#             except OSError as e:
#                 print(e)
#                 print(filename)
#
#             sleep(0.5)