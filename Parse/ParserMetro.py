# импорт бибилиотеки
from bs4 import BeautifulSoup
import requests
import pandas as pd
#  список данных
data = []
# без цикла
page = requests.get('https://m.101hotels.com/recreation/russia/moskva/infrastructure/metro?ysclid=la3a4g68is638416669&page=23')
# проверка статуса страниц 
if page.status_code != 200:
    print(page.status_code)
    exit()
# индексация информации по выбранным тегам в блоки
soup = BeautifulSoup(page.text, features="html.parser")
metros = soup.find_all('li', class_='tiled_list_item')  
for metro in metros:
    block_01 = metro.find('div', class_='link__title')
    block_02 = metro.find('div', class_='center-distance')
# перевод блоков в строки и обработка строк
    name = block_01.text.strip().replace('\n', '\t')
    center_distance = block_02.text.strip().replace('\n', '\t')
# добавление в список данных
    data.append({
      'metro_name': name,
      'center_distance': center_distance,
            })
# выходной выйл
df = pd.DataFrame(data).to_csv('metro.csv',index=False)