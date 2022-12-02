import pandas as pd
import requests
import numpy as np
flat =pd.read_csv('E:\project university\Выпускная работа\DataSet\Недвижка\my_DS\data_all.csv')
flat[['rooms', 'square','floor1']] = flat['description']. str.split('|', 0 , expand= True )# разделяю значения по столбцам
flat['all_info'] =flat.apply(lambda x: x['all_info'].replace(x['address'],''), axis=1) # удаляю из all_info ифу по address
flat['all_info'] =flat.apply(lambda x: x['all_info'].replace(x['description'],''), axis=1) # удаляю из all_info ифу по description
flat = flat.drop(columns='description') #удаляю колонку старого дескрипшина
flat[['des_property']] = flat['all_info'].str.extract('(?:(?<=пешком)|(?<=транспортом))(.*)', 0, expand= True) #добавляю новый дескрипшн
flat.des_property.fillna(flat.all_info, inplace=True) # копирую описание из all_info в ячейки NAN
flat['all_info'] =flat.apply(lambda x: x['all_info'].replace(x['des_property'],''), axis=1) # удаляю из all_info ифу по property и получаю метро
flat.rename(columns={'all_info': 'metro','des_property': 'description'},inplace = True )# переименовываю столбцы
flat = flat.drop(columns='Unnamed: 0') 
flat['cost_for_meter(₽)'] =((flat['cost'])/(flat['square'].str.extract(r'(.*(?=\м²))', expand=False).astype("float"))).astype("float") # получаю еще один столбец с ценой квадратного метра
flat['cost_for_meter(₽)'] = round(flat['cost_for_meter(₽)'],1) #округляю до десятых
flat['cost']=flat['cost'].astype("int") 
flat.rename(columns={'cost': 'cost(₽)'},inplace = True )# переименовываю столбцы
                                                  # Перевожу пешком и на транспорте в метры от метро и очищаю от букв
flat[['metro_name', 'min']] = flat['metro'].str.split(',' , 1 , expand = True)# Отделяю минуты и название метро
flat['metro_name'] = flat['metro_name'].str.lower()# Перевод названий станций в нижний регистр
flat['min_int'] =flat['min'].str.extract(r'(\d+(?= \мин.))').fillna(0) #Выделяю только цифры
flat['min_int'] =flat['min_int'].astype("int")#перевожу в INT
flat['alpha'] = flat['min'].str.contains('пешком') # нахожу слова -пешком истина -есть ложь нет
flat['alpha']=flat['alpha'].replace({False:333,	True:84}).fillna(0).astype("int") # меняю ложь на 333 м/мин (20км/ч) и истину на 84 м/мин(5км/ч), и перевожу в INT
flat['distance(м)']= flat['alpha']*flat['min_int'] # перемножаю минуты на скорость получаю метры до метро
flat['distance(м)']=flat['distance(м)'].astype("float") # перевожу расстояние до метро в тип флоат
flat[['floor2']] = flat['floor1'].str.extract('(?:(?<=этаж))(.*)', 0, expand= True)# удаляю слово этаж
flat[['floor','floors']] = flat['floor2'].str.split('/', 0 , expand= True )# разделяю значения этажей по столбцам
flat['floor']=flat['floor'].astype("int") # перевожу этаж в тип INT
flat['floors']=flat['floors'].astype("int") # перевожу этажи в тип INT
flat[['square(м²)','DEL']] = flat['square'].str.split('м', 0 , expand= True ) # удаляю ненужные м²
flat['square(м²)']=flat['square(м²)'].astype("float")# перевожу площадь в тип флоат
flat = flat.drop(['alpha','min_int','min','metro','floor1','floor2','square','DEL'],axis=1) # удаляю промежуточные столбцы для вычислений
flat = flat[['rooms','cost(₽)','square(м²)','cost_for_meter(₽)','address','metro_name','distance(м)','floor','floors','description']]#порядок столбцов
#Удаляю строки с неверными данными
flat=flat.drop(axis=1, index=31397)
flat=flat.drop(axis=1, index=31401)
flat=flat.drop(axis=1, index=646)
flat.to_csv('E:\project university\Выпускная работа\DataSet\Недвижка\my_DS\moscow_flat22.csv', index=False)
flat.head(5)