import pandas as pd
import requests
flat_full =pd.read_csv('E:\project university\Выпускная работа\DataSet\Недвижка\my_DS\moscow_full_2.0.csv')
flat_full[['rooms', 'square','floor1']] = flat_full['title']. str.split(', ', 0 , expand= True )# разделяю значения по столбцам
flat_full.rename(columns = {'adress' : 'address'}, inplace = True) # меняю наименование на корректное
flat_full = flat_full.drop(columns='title') #удаляю колонку старого title
flat_full[['floor2','floor3']] = flat_full['floor1'].str.split('э', 0 , expand= True ) # разделяю floor1 значения по столбцам
flat_full[['floor', 'floors']] = flat_full['floor2']. str.split('/', 0 , expand= True ).fillna(0)# разделяю floor2 значения по столбцам
flat_full['floor']=flat_full['floor'].astype("int") # перевожу этаж в тип INT
flat_full['floors']=flat_full['floors'].astype("int") # перевожу этажи в тип INT
flat_full[['square(м²)','DEL']] = flat_full['square'].str.split('м', 0 , expand= True ) # удаляю ненужные м²
flat_full['square(м²)']=flat_full['square(м²)'].astype("float")# перевожу площадь в тип флоат
                               # Очищаю cost_for_meter(₽)
flat_full[['cost_for_meter(₽)', 'DEL1']] = flat_full['cost_for_meter']. str.split(' ₽ ', 0 , expand= True )# разделяю значения по столбцам
flat_full['cost_for_meter(₽)']=flat_full['cost_for_meter(₽)'].fillna(0)# меняю нан на 0
mask = flat_full['cost_for_meter(₽)'].str.fullmatch(r'[\d ]*')# ввожу переменную с регулярным выражением
flat_full.loc[mask, 'cost_for_meter(₽)'] = flat_full.loc[mask, 'cost_for_meter(₽)'].str.replace(' ', '')# меняю replace ом пробел
flat_full['cost_for_meter(₽)']=flat_full['cost_for_meter(₽)'].astype("int")# перевожу в флоат
flat_full['cost_for_meter(₽)'] = (flat_full['cost_for_meter(₽)'] + ((flat_full['cost_for_meter(₽)'])*(34/100)))
                               # Очищаю cost
flat_full[['cost(₽)', 'DEL2']] = flat_full['cost']. str.split('₽', 0 , expand= True )# разделяю значения по столбцам
flat_full['cost(₽)']=flat_full['cost(₽)'].fillna(0)# меняю нан на 0
mask = flat_full['cost(₽)'].str.fullmatch(r'[\d ]*')# ввожу переменную с регулярным выражением
flat_full.loc[mask, 'cost(₽)'] = flat_full.loc[mask, 'cost(₽)'].str.replace(' ', '')# меняю replace ом пробел
flat_full['cost(₽)']=flat_full['cost(₽)'].astype("float")# перевожу в float
flat_full['cost(₽)'] = (flat_full['cost(₽)'] + ((flat_full['cost(₽)'])*0.34)) # увеличиваю на 34%

                                # Получаю расстояние в метрах
flat_full['min_int']= flat_full['metro'].str.extract(r'([-+]?(?:\d+(?:\.\d*)?|\.\d+))(?= м| км)').fillna(0) #Выделяю только цифры
flat_full['min_int']=flat_full['min_int'].astype("float")# перевожу в INT
flat_full['alpha1'] = flat_full['metro'].str.contains(' км') # нахожу слова -км истина -есть ложь нет
flat_full['alpha1']=flat_full['alpha1'].replace({False:1,	True:1000}).fillna(0).astype("int") # меняю ложь на 1  и истину на 1000 (в 1 км =1000м), и перевожу в INT
flat_full['distance(м)']= flat_full['alpha1']*flat_full['min_int']  # получаю расстояние от метров метрах
                                # выделяю название метро
flat_full['metro_name']= flat_full['metro'].str.extract(r'(.*(?= \d))') # выделяю название метро
flat_full['metro_name']= flat_full['metro_name'].str.lower() # привожу к нижнему регистру
                                # удаляю промежуточные колонки
flat_full = flat_full.drop(['floor3','floor2','floor1','DEL','square', 'cost_for_meter', 'DEL1','cost','DEL2','min_int','alpha1','metro'],axis=1)  #удаляю промежуточные колонки
flat_full= flat_full[['rooms','cost(₽)','square(м²)','cost_for_meter(₽)','address','metro_name','distance(м)','floor','floors','description']]#порядок столбцов
flat_full.to_csv('E:\project university\Выпускная работа\DataSet\Недвижка\my_DS\moscow_flat20.csv',index=False)