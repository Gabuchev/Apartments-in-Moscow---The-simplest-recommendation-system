from bs4 import BeautifulSoup
import requests
import pandas as pd
data = []
for i in range(1,2306,1):
    url = 'https://www.egsnk.ru/kvartiry/kupit/p'
    url = url + str(i)
    page = requests.get(url)
    if page.status_code != 200:
        print(page.status_code)
        exit()
    soup = BeautifulSoup(page.text, features="html.parser")
    flats = soup.find_all('div', class_='res_row clearfix')  
    for flat in flats:
        block_01 = flat.find('div', class_='res_right')
        block_02_01 = flat.find_all('div', class_='mt8')
        block_02_02 = block_02_01[0]
        block_03 = flat.find('div', class_='res_text fs15')
        block_03_02 = block_03.find_all('div')
        block_03_03 = block_03_02[0]
            
        text_01 = block_01.text.strip().replace('\n', '\t')
        cost = float(''.join([x for x in text_01 if x.isdigit() or x == '.']))
        description = block_03_03.text.strip().replace('\n', '\t')
        info = block_03.text.strip().replace('\n', '\t')
        address = block_02_02.text.strip().replace('\n', '\t')

        data.append({
                'cost': cost,
                'address': address,
                'description': description,
                'all_info': info,
            })
df = pd.DataFrame(data).to_csv('data_all.csv')