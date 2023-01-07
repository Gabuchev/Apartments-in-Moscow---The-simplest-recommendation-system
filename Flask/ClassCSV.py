import pandas as pd  
import pickle 
import re
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing

from gensim.models.doc2vec import TaggedDocument
###################################################################################################################################    
class DataCSV:

   def __init__(self, Cost, Square, Distance, Floor, Floors, Description, MetroName,CenterDistance):
      
      self.Cost  = Cost 
      self.Square = Square
      self.Distance = Distance
      self.Floor  = Floor 
      self.Floors = Floors
      self.Description = Description
      self.CenterDistance = CenterDistance
      self.MetroName = MetroName

#Запаолняю дата-лист

      DataList = []
      DataList.append({'cost(₽)': self.Cost,'square(м²)': self.Square, 'distance(м)': self.Distance, "center_distance/m": self.CenterDistance,"floor": self.Floor,
      "floors":self.Floors, "description":self.Description,"metro_name":self.MetroName})

#Сохраняю даталист в виде фоейма данных НА ДАЛЬНЕЙШУЮ ОБРАБОТКУ

      df = pd.DataFrame(DataList).to_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\DFInput.csv', index=False) #  **********  DFInput.csv  ********** 

      df = pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\DFInput.csv')  # ЭТО НАДО!

      #Получаю столбец с ценой квадратного метра

      df['cost_for_meter(₽)'] =round((df['cost(₽)'])/(df['square(м²)']).astype("float"),1)  # получаю столбец с ценой квадратного метра
      df = df[['cost(₽)','square(м²)','cost_for_meter(₽)','metro_name','distance(м)','center_distance/m','floor','floors','description']]#порядок столбцов
      df['floors']=df['floors'].astype("int")# перевожу Этажность дома в тип int
      df['floor']=df['floor'].astype("int")# перевожу Этаж в тип int

      df.to_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\DFInput.csv',index=False)

      #df_1 = pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\DFInput.csv')
      #self.df = df_1 # ЗДЕСЬ ПРОВЕРИТЬ ЗНАЧЕНИЕ ПЕРЕДАЕТСЯ ДАЛЕЕ ИЛИ ОСТАЕТСЯ КАК ЕСТЬ

                # ТАБЛИЦА ВВЕДЕННЫХ ДАННЫХ

      df.rename(columns = {'cost(₽)' : 'Цена','square(м²)' : 'Площадь','cost_for_meter(₽)' : 'Цена за кв.м','metro_name' : 'Название метро', 
      'distance(м)' : 'Расстояние до метро','center_distance/m' : 'Расстояние до центра','floor' : 'Этаж',
      'floors' : 'Этажность дома','description' : 'Описание'}, inplace = True) # меняю наименование на корректное
      df.to_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\DFTableInput.csv',index=False) # **********  DFTableInput.csv  ********** 
      
###################################################################################################################################    


   def RecAll (self): #Обрабатываю слова 

        df_1 = pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\DFInput.csv')
        nlp = pickle.load(open(r"C:\Users\Daniel\project university\Csv_to_html\PKL\nlppickle.pkl", "rb"))
        def cleaning(doc):
            txt = [token.lemma_ for token in doc]
            return ' '.join(txt)
          #Удаляю неалфавитные символы:
        brief_cleaning = (re.sub("[^А-Яа-я]", ' ', str(row)).lower() for row in df_1['description'])
        txt = [cleaning(doc) for doc in nlp.pipe(brief_cleaning, batch_size=1000, n_process=1)]

        #Помещаю результаты в фрейм данных
        dfD = pd.DataFrame({'description': txt})
        #Удаляю пробелы которых больше или равно 2 
        dfD = dfD['description'].str.replace(' {2,}', ' ', regex=True)
                        # ДАТАФРЕЙМ С ОПИСАНИЕМ ДЛЯ ВВЕДЕННЫХ ДАННЫХ
        dfD.to_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\DFDescription.csv',index=False) # **********  DFDescription.csv  **********
        
        ###################################################################################################################################

        DDf= pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\DFDescription.csv')   # НАДО!     

        #Загружаю модель doc2vec
        loaded_model = pickle.load(open(r"C:\Users\Daniel\project university\Csv_to_html\PKL\d2vpickle.pkl", "rb"))       
        #Генерация векторов
        doc2vec = [loaded_model.infer_vector((DDf['description'][i].split(' '))) for i in range(0,len(DDf['description']))]
        Vec64 = pd.DataFrame(data = doc2vec)  

        #Vec64.to_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\Vec64_Des_Input.csv',index=False)  # ДАТАСЕТ ДЛЯ ПРОВЕРКИ (УДАЛИТЬ ПОСЛЕ ОБРАБОТКИ)

        #КОНКАТЕНАЦИЯ (получаю набор данных с векторами по описанию)

        Data = pd.concat([df_1, Vec64], axis=1)
        Data = Data.drop(['description'], axis=1)#Удаляю столбец с описанием  

        ###################################################################################################################################

        #ДАТА СЕТ ДЛЯ ВВЕДЕННЫХ ДАННЫХ БЕЗ НОРМАЛИЗАЦИИ БЕЗ МАСШТАБИРОВАНИЯ БЕЗ ЗНАЧЕНИЯ МЕТРО

        Data.to_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\DFProcess.csv',index=False)     # **********  DFProcess.csv  **********

        #Считываю данные из Metro_encoders.csv

        Metro_Encoder_all = pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\Metro_encoders.csv')

        #Сопоставлю каждому названию метро свой код из Metro_encoders.csv

        Metro_Encoder = Metro_Encoder_all[Metro_Encoder_all['metro_name'] == self.MetroName]["Metro_encoders"].values

        #Metro_Encoder=str(Metro_Encoder).strip('[]')
        Metro_encoders_Input = pd.DataFrame(data = Metro_Encoder).astype("int")
        Metro_encoders_Input.rename(columns = {0: "Metro_encoders"}, inplace = True) #ПЕРЕИМЕНОВЫВАЮ СТОЛБЕЦ НА "Metro_encoders"

        #КОНКАТЕНАЦИЯ

        DData=pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\DFProcess.csv')
        Data_all = pd.concat([DData, Metro_encoders_Input], axis=1) #ДАННЫЕ INPUT ДОБАВЛЯЮ В КОНЕЦ ДАТАСЕТА

        #удаляю metro_name
        Data_all = Data_all.drop(['metro_name'],axis=1)

        #Data_all.to_csv(r'C:\Users\Daniel\project university\Csv_to_html\DData2.csv',index=False)

        ###################################################################################################################################

        #Нормировка:
        #Data_all=pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\DData2.csv')

        data1 = pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\Data2.csv') #ЗНАЧЕНИЯ ДАННЫХ ВСЕГО ДАТАСЕТА

        #КОНКАТЕНАЦИЯ
        frame = [data1 , Data_all] #ДАННЫЕ INPUT ДОБАВЛЯЮ В КОНЕЦ ДАТАСЕТА
        data2 = pd.concat(frame)

        data2.to_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\DFProcess2.csv',index=False) # **********    DFProcess2    **********

        #МАСШТАБИРОВАНИЕ

        data2 = pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\DFProcess2.csv')
        scaler = preprocessing.MinMaxScaler()
        dataNorm_ALL = scaler.fit_transform(data2)

        dataNorm_PD = pd.DataFrame(data = dataNorm_ALL,columns = data2.columns.values)     

        csr_Ddata = csr_matrix(dataNorm_PD.values)

        recommendations = 10

        # Загружаю модель knn
        #knn = pickle.load(open(r"C:\Users\Daniel\project university\Csv_to_html\knnpickle.pkl", "rb"))
        knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute', n_neighbors = 10, n_jobs = -1)

        #обучаю модель

        knn.fit(csr_Ddata)
        distances, indices = knn.kneighbors(csr_Ddata[-1], n_neighbors = recommendations + 1)

        indices_list = indices.squeeze().tolist()
        distances_list = distances.squeeze().tolist()
        indices_distances = list(zip(indices_list, distances_list))    

        ###################################################################################################################################

        indices_distances_sorted = sorted(indices_distances, key = lambda x: x[1], reverse = False)

        indices_distances_sorted = indices_distances_sorted[1:]  

        dataСlean = pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\Сlean.csv',encoding='utf-8')

        DFComplete = pd.DataFrame(list(indices_distances_sorted), columns=["id","Cos_dist"])
        
        recom_df = pd.merge(DFComplete, dataСlean, how='left',on='id')  

        recom_df.rename(columns = {'cost(₽)' : 'Cost','square(м²)' : 'Square','cost_for_meter(₽)' : 'CostForMeter',
        'distance(м)' : 'Distance','center_distance/m' : 'CenterDistance','floor' : 'Floor',
        'floors' : 'Floors','description' : 'Description'}, inplace = True)
        
        recom_df.to_json(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\recom_df.json')
   
###################################################################################################################################

   def RecCost (self):

        #Загружаю данные, удаляю описание

        df_2 = pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\DFInput.csv')
        df_2 = df_2.drop(['description'],axis=1)
        
        ###################################################################################################################################

        #ДАТА СЕТ ДЛЯ ВВЕДЕННЫХ ДАННЫХ БЕЗ НОРМАЛИЗАЦИИ БЕЗ МАСШТАБИРОВАНИЯ БЕЗ ЗНАЧЕНИЯ МЕТРО

        df_2.to_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\DFProcessCost.csv',index=False)    # **********  DFProcessCost.csv  **********

        #Считываю данные из Metro_encoders.csv

        Metro_Encoder_all = pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\Metro_encoders.csv')

        #Сопоставлю каждому названию метро свой код из Metro_encoders.csv

        Metro_Encoder = Metro_Encoder_all[Metro_Encoder_all['metro_name'] == self.MetroName]["Metro_encoders"].values

        #Metro_Encoder=str(Metro_Encoder).strip('[]')
        Metro_encoders_Input = pd.DataFrame(data = Metro_Encoder).astype("int")
        Metro_encoders_Input.rename(columns = {0: "Metro_encoders"}, inplace = True) #ПЕРЕИМЕНОВЫВАЮ СТОЛБЕЦ НА "Metro_encoders"

        #КОНКАТЕНАЦИЯ

        DData=pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\DFProcessCost.csv')
        Data_all = pd.concat([DData, Metro_encoders_Input], axis=1) #ДАННЫЕ INPUT ДОБАВЛЯЮ В КОНЕЦ СТРОКИ ДАТАСЕТА axis=1

        #удаляю metro_name
        Data_all = Data_all.drop(['metro_name'],axis=1)

        #Data_all.to_csv(r'C:\Users\Daniel\project university\Csv_to_html\DData2.csv',index=False)

        ###################################################################################################################################

        #Нормировка:
        #Data_all=pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\DData2.csv')

        data3 = pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\Data2.csv') #ЗНАЧЕНИЯ ДАННЫХ ВСЕГО ДАТАСЕТА
        data3 = data3[['cost(₽)', 'square(м²)', 'cost_for_meter(₽)', 'distance(м)','center_distance/m', 'floor','floors', 'Metro_encoders']]

        #КОНКАТЕНАЦИЯ
        frame = [data3 , Data_all] #ДАННЫЕ INPUT ДОБАВЛЯЮ В КОНЕЦ ДАТАСЕТА
        data4 = pd.concat(frame)

        data4.to_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\DFProcessCost2.csv',index=False) # **********    DFProcessCost2.csv   **********

        #МАСШТАБИРОВАНИЕ

        data4 = pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\DFProcessCost2.csv')
        scaler = preprocessing.MinMaxScaler()
        dataNorm_ALL = scaler.fit_transform(data4)

        dataNorm_PD = pd.DataFrame(data = dataNorm_ALL,columns = data4.columns.values)       
        csr_Ddata1 = csr_matrix(dataNorm_PD.values)

        #АЛГОРИТМ - KNN 

        recommendations = 10

        # Загружаю модель knn
        #knn = pickle.load(open(r"C:\Users\Daniel\project university\Csv_to_html\knnpickle.pkl", "rb"))
        knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute', n_neighbors = 10, n_jobs = -1)

        #обучаю модель

        knn.fit(csr_Ddata1)
        distances, indices = knn.kneighbors(csr_Ddata1[-1], n_neighbors = recommendations + 1) # -1 потому что погследний элемент добавлен конкатенацией

        indices_list = indices.squeeze().tolist()
        distances_list = distances.squeeze().tolist()
        indices_distances = list(zip(indices_list, distances_list))

        ###################################################################################################################################

        indices_distances_sorted = sorted(indices_distances, key = lambda x: x[1], reverse = False)

        indices_distances_sorted = indices_distances_sorted[1:]  

        dataСlean = pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\Сlean.csv',encoding='utf-8')
                
        DFComplete = pd.DataFrame(list(indices_distances_sorted), columns=["id","Cos_dist"])

        #DFComplete.to_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\DFComplete.csv',index=False)  # ДАТАСЕТ ДЛЯ ПРОВЕРКИ (УДАЛИТЬ ПОСЛЕ ОБРАБОТКИ)
      
        recom_df = pd.merge(DFComplete, dataСlean, how='left', on='id')  

        recom_df.rename(columns = {'cost(₽)' : 'Cost','square(м²)' : 'Square','cost_for_meter(₽)' : 'CostForMeter',
        'distance(м)' : 'Distance','center_distance/m' : 'CenterDistance','floor' : 'Floor',
        'floors' : 'Floors','description' : 'Description'}, inplace = True)
        
        recom_df.to_json(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\recom_df.json')
   
###################################################################################################################################

   def RecDescription (self):

        df_Description = pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\DFInput.csv')
        df_Description = df_Description[['description']]
        nlp = pickle.load(open(r"C:\Users\Daniel\project university\Csv_to_html\PKL\nlppickle.pkl", "rb"))

        def cleaning(doc):
            #txt = [token.lemma_ for token in doc if not token.is_stop]
            txt = [token.lemma_ for token in doc]
            return ' '.join(txt)
          #Удаляю неалфавитные символы:
        brief_cleaning = (re.sub("[^А-Яа-я]", ' ', str(row)).lower() for row in df_Description['description'])
        txt = [cleaning(doc) for doc in nlp.pipe(brief_cleaning, batch_size=1000, n_process=1)]



        #Помещаю результаты в фрейм данных
        dfD = pd.DataFrame({'description': txt})
        #Удаляю пробелы которых больше или равно 2 
        dfD = dfD['description'].str.replace(' {2,}', ' ', regex=True)
                        # ДАТАФРЕЙМ С ОПИСАНИЕМ ДЛЯ ВВЕДЕННЫХ ДАННЫХ
        dfD.to_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\DF-DefRecDes.csv',index=False) # **********  DF-DefRecDes.csv  **********

        ###################################################################################################################################
        
        DDf= pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\DF-DefRecDes.csv')   # НАДО! 

        #Загружаю модель doc2vec
        loaded_model = pickle.load(open(r"C:\Users\Daniel\project university\Csv_to_html\PKL\d2vpickle.pkl", "rb")) 

        #Генерация векторов
        doc2vec = [loaded_model.infer_vector((DDf['description'][i].split(' '))) for i in range(0,len(DDf['description']))]

        Vec64 = pd.DataFrame(data = doc2vec) 

        Vec64.to_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\Vec64_Des_Input.csv',index=False)  # ДАТАСЕТ ДЛЯ ПРОВЕРКИ (УДАЛИТЬ ПОСЛЕ ОБРАБОТКИ)

        df_Vec = pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\Vec64.csv') 

        Vec64= pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\Vec64_Des_Input.csv') 

        #КОНКАТЕНАЦИЯ (получаю набор данных с векторами по описанию)

        frame = [df_Vec , Vec64] #ДАННЫЕ INPUT ДОБАВЛЯЮ В КОНЕЦ ДАТАСЕТА

        Data = pd.concat(frame)

        Data.to_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\Data_Des_Input.csv',index=False)
        
        Data =pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\Data_Des_Input.csv') # **********  Data_Des_Inpu.csv  **********

        ###################################################################################################################################

        csr_Ddata2 = csr_matrix(Data.values)

        #АЛГОРИТМ - KNN 

        recommendations = 10

        # Загружаю модель knn
        #knn = pickle.load(open(r"C:\Users\Daniel\project university\Csv_to_html\knnpickle.pkl", "rb"))
        knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute', n_neighbors = 10, n_jobs = -1)

        #обучаю модель

        knn.fit(csr_Ddata2)
        distances, indices = knn.kneighbors(csr_Ddata2[-1], n_neighbors = recommendations + 1) # -1 потому что погследний элемент добавлен конкатенацией

        indices_list = indices.squeeze().tolist()
        distances_list = distances.squeeze().tolist()
        indices_distances = list(zip(indices_list, distances_list))

        ###################################################################################################################################

        indices_distances_sorted = sorted(indices_distances, key = lambda x: x[1], reverse = False)

        indices_distances_sorted = indices_distances_sorted[1:]  

        dataСlean = pd.read_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\Сlean.csv',encoding='utf-8')
                
        DFComplete = pd.DataFrame(list(indices_distances_sorted), columns=["id","Cos_dist"])

        #DFComplete.to_csv(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\DFComplete.csv',index=False)  # ДАТАСЕТ ДЛЯ ПРОВЕРКИ (УДАЛИТЬ ПОСЛЕ ОБРАБОТКИ)
      
        recom_df = pd.merge(DFComplete, dataСlean, how='left', on='id')  

        recom_df.rename(columns = {'cost(₽)' : 'Cost','square(м²)' : 'Square','cost_for_meter(₽)' : 'CostForMeter',
        'distance(м)' : 'Distance','center_distance/m' : 'CenterDistance','floor' : 'Floor',
        'floors' : 'Floors','description' : 'Description'}, inplace = True)
        
        recom_df.to_json(r'C:\Users\Daniel\project university\Csv_to_html\ClassSave\recom_df.json')       
       
